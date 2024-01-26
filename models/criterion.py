import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits, l1_loss, mse_loss
from torch.distributed import all_reduce
from torchvision.ops.boxes import nms
import math
from scipy.optimize import linear_sum_assignment

from utils.box_utils import box_cxcywh_to_xyxy, generalized_box_iou
from utils.distributed_utils import is_dist_avail_and_initialized, get_world_size
from collections import defaultdict


class HungarianMatcher(nn.Module):

    def __init__(self,
                 coef_class: float = 2,
                 coef_bbox: float = 5,
                 coef_giou: float = 2):
        super().__init__()
        self.coef_class = coef_class
        self.coef_bbox = coef_bbox
        self.coef_giou = coef_giou
        assert coef_class != 0 or coef_bbox != 0 or coef_giou != 0, "all costs cant be 0"

    def forward(self, pred_logits, pred_boxes, annotations):
        with torch.no_grad():
            bs, num_queries = pred_logits.shape[:2]
            # We flatten to compute the cost matrices in a batch
            pred_logits = pred_logits.flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_class]
            pred_boxes = pred_boxes.flatten(0, 1)  # [batch_size * num_queries, 4]
            gt_class = torch.cat([anno["labels"] for anno in annotations]).to(pred_logits.device)
            gt_boxes = torch.cat([anno["boxes"] for anno in annotations]).to(pred_logits.device)
            # Compute the classification cost.
            alpha, gamma = 0.25, 2.0
            neg_cost_class = (1 - alpha) * (pred_logits ** gamma) * (-(1 - pred_logits + 1e-8).log())
            pos_cost_class = alpha * ((1 - pred_logits) ** gamma) * (-(pred_logits + 1e-8).log())
            cost_class = pos_cost_class[:, gt_class] - neg_cost_class[:, gt_class]
            # Compute the L1 cost between boxes
            cost_boxes = torch.cdist(pred_boxes, gt_boxes, p=1)
            # Compute the giou cost between boxes
            cost_giou = - generalized_box_iou(box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(gt_boxes))
            # Final cost matrix
            cost = self.coef_bbox * cost_boxes + self.coef_class * cost_class + self.coef_giou * cost_giou
            cost = cost.view(bs, num_queries, -1).cpu()
            sizes = [len(anno["boxes"]) for anno in annotations]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class SetCriterion(nn.Module):

    def __init__(self,
                 num_classes=9,
                 coef_class=2,
                 coef_boxes=5,
                 coef_giou=2,
                 coef_domain=1.0,
                 coef_domain_bac=0.3,
                 alpha_focal=0.25,
                 device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.matcher = HungarianMatcher()
        self.coef_class = coef_class
        self.coef_boxes = coef_boxes
        self.coef_giou = coef_giou
        self.coef_domain = coef_domain
        self.coef_domain_bac = coef_domain_bac
        self.alpha_focal = alpha_focal
        self.logits_sum = [torch.zeros(1, dtype=torch.float, device=device) for _ in range(num_classes)]
        self.logits_count = [torch.zeros(1, dtype=torch.int, device=device) for _ in range(num_classes)]

    @staticmethod
    def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
        Returns:
            Loss tensor
        """
        prob = inputs.sigmoid()
        ce_loss = binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean(1).sum() / num_boxes

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_class(self, pred_logits, annotations, indices, num_boxes):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(annotations, indices)]) # there may occur format problems
        target_classes = torch.full(pred_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=pred_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([pred_logits.shape[0], pred_logits.shape[1], pred_logits.shape[2] + 1],
                                            dtype=pred_logits.dtype, layout=pred_logits.layout, device=pred_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = self.sigmoid_focal_loss(pred_logits, target_classes_onehot, num_boxes, alpha=self.alpha_focal, gamma=2) * pred_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        return losses

    def loss_boxes(self, pred_boxes, annotations, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        idx = self._get_src_permutation_idx(indices)
        src_boxes = pred_boxes[idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(annotations, indices)], dim=0)

        loss_bbox = l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {'loss_bbox': loss_bbox.sum() / num_boxes}

        return losses

    def loss_giou(self, pred_boxes, annotations, indices, num_boxes):

        idx = self._get_src_permutation_idx(indices)
        src_boxes = pred_boxes[idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(annotations, indices)], dim=0)

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        losses = {'loss_giou': loss_giou.sum() / num_boxes}
 
        return losses

    def loss_domains(self, out, domain_label):
        # loss_dict = {}
        # pred_logits = out['pred_logits']
        # pred_boxes = out['pred_boxes']
        # for key in record_dict.keys():
        #     if key.startswith("loss"):
        #         if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo":
        #             # pseudo bbox regression <- 0
        #             loss_dict[key] = record_dict[key] * 0
        #         elif key[-6:] == "pseudo":  # unsupervised loss
        #             loss_dict[key] = (
        #                 record_dict[key] *
        #                 self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
        #             )
        #         elif (
        #             key == "loss_D_img_s" or key == "loss_D_img_t"
        #         ):  # set weight for discriminator
        #             # import pdb
        #             # pdb.set_trace()
        #             loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.DIS_LOSS_WEIGHT #Need to modify defaults and yaml
        #         else:  # supervised loss
        #             loss_dict[key] = record_dict[key] * 1

        # losses = sum(loss_dict.values())   
        # losses = {'loss_domain': loss_domains.sum() / num_boxes}     
        # return losses
        pass

    def forward(self, out, annotations=None, domain_label=None, enable_mae=False):
        # Implement here
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(out['pred_logits'], out['pred_boxes'], annotations)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in annotations)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(out.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        loss_dict = {}
 
        loss_dict.update(self.loss_class(out['pred_logits'], annotations, indices, num_boxes))
        loss_dict.update(self.loss_boxes(out['pred_boxes'], annotations, indices, num_boxes))
        loss_dict.update(self.loss_giou(out['pred_boxes'], annotations, indices, num_boxes))
        loss_dict.update(self.loss_domains(out['loss_domain'], annotations, indices, num_boxes))
        #######################################

        loss = self.coef_class * loss_dict['loss_ce'] + self.coef_boxes * loss_dict['loss_bbox'] + \
            self.coef_giou * loss_dict['loss_giou'] + self.coef_domain * loss_dict['loss_domain'] #+ self.coef_domain_bac * ????

        return loss, loss_dict


@torch.no_grad()
def post_process(pred_logits, pred_boxes, image_sizes, topk=100):
    # print(f"pred_logits: {pred_logits.shape}")
    # print(f"pred_boxes: {pred_boxes.shape}")
    # print(f"image_sizes: {image_sizes.shape}")
    assert len(pred_logits) == len(image_sizes)
    assert image_sizes.shape[1] == 2
    prob = pred_logits.sigmoid()
    prob = prob.view(pred_logits.shape[0], -1)
    topk_values, topk_indexes = torch.topk(prob, topk, dim=1)
    topk_boxes = torch.div(topk_indexes, pred_logits.shape[2], rounding_mode='trunc')
    labels = topk_indexes % pred_logits.shape[2]
    boxes = box_cxcywh_to_xyxy(pred_boxes)
    boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
    # From relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = image_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]
    results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(topk_values, labels, boxes)]
    return results


def get_pseudo_labels(pred_logits, pred_boxes, thresholds, nms_threshold=0.7):
    probs = pred_logits.sigmoid()
    scores_batch, labels_batch = torch.max(probs, dim=-1)
    pseudo_labels = []
    thresholds_tensor = torch.tensor(thresholds, device=pred_logits.device)
    for scores, labels, pred_box in zip(scores_batch, labels_batch, pred_boxes):
        larger_idx = torch.gt(scores, thresholds_tensor[labels]).nonzero()[:, 0]
        scores, labels, boxes = scores[larger_idx], labels[larger_idx], pred_box[larger_idx, :]
        nms_idx = nms(box_cxcywh_to_xyxy(boxes), scores, iou_threshold=nms_threshold)
        scores, labels, boxes = scores[nms_idx], labels[nms_idx], boxes[nms_idx, :]
        pseudo_labels.append({'scores': scores, 'labels': labels, 'boxes': boxes})
    return pseudo_labels
