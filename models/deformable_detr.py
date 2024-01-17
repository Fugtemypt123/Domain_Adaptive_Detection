import math
import copy

import torch
from torch.nn.functional import relu, interpolate
from torch import nn


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MultiConv2d(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Conv2d(n, k, kernel_size=(3, 3), padding=1) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MultiConv1d(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Conv1d(n, k, kernel_size=3) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, eta=1.0):
        ctx.eta = eta
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output * -ctx.eta), None


def grad_reverse(x, eta=1.0):
    return GradReverse.apply(x, eta)

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


class DeformableDETR(nn.Module):

    def __init__(self, backbone, position_encoding, transformer, num_classes, num_queries, num_feature_levels):
        super().__init__()
        self.backbone = backbone
        self.position_encoding = position_encoding
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.hidden_dim
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        self.query_embed = nn.Embedding(num_queries, hidden_dim*2)

        num_backbone_outs = backbone.num_outputs
        input_proj_list = []
        for _ in range(num_backbone_outs):
            in_channels = backbone.get_backbone()[1][_]
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
            ))
        for _ in range(num_feature_levels - num_backbone_outs):
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(32, hidden_dim),
            ))
            in_channels = hidden_dim
        self.input_proj = nn.ModuleList(input_proj_list)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = transformer.decoder.num_layers

        nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
        self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
        self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
        self.transformer.decoder.bbox_embed = None
    
    def forward(self, x, ms):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        # backbone and mask
        features = self.backbone(x)
        masks = []
        for x in features:
            m = ms
            assert m is not None
            masks.append(interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0])

        # positional encoding
        pos = []
        for l in range(len(features)):
            pos.append(self.position_encoding(features[l], masks[l]).to(features[l].dtype))

        # input projection
        srcs = []
        for l in range(len(features)):
            srcs.append(self.input_proj[l](features[l]))
    
        # add last feature level
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1])
                else:
                    src = self.input_proj[l](srcs[-1])
                m = ms
                mask = interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.position_encoding(src, mask).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = self.query_embed.weight
        hs, init_reference, inter_references = self.transformer(srcs, masks, pos, query_embeds)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            # print(f"lvl:{lvl}, outputs_bbox:{tmp}")
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
                # print(f"lvl:{lvl}, tmp.shape:{tmp.shape}")
            outputs_coord = tmp.sigmoid()
            # print(f"lvl:{lvl}, outputs_class:{outputs_class}")
            
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        # print(outputs_class.shape)
        # print(outputs_coord.shape)
        out = {'logits_all': outputs_class, 'boxes_all': outputs_coord, 'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        # print(out['pred_logits'].shape)
        # print(out['pred_boxes'].shape)
        return out


