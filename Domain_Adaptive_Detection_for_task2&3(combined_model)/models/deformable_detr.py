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
        self.backbone = backbone # backbone,这里是resnet50
        self.position_encoding = position_encoding # position_encoding,具体参见同目录下对应文件
        self.num_queries = num_queries # object_queries的数量，默认是300
        self.transformer = transformer # transformer模块
        hidden_dim = transformer.hidden_dim # 隐藏层，默认256
        self.class_embed = nn.Linear(hidden_dim, num_classes) # 最终产生分类的投影层
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3) # 最终产生框的投影层，总共3层拼在一起
        self.num_feature_levels = num_feature_levels # 使用的特征图层数，默认是4
        self.query_embed = nn.Embedding(num_queries, hidden_dim*2) # object_queries可学习的那个embedding

        num_backbone_outs = backbone.num_outputs # backbone能返回多少层，这个值是3
        input_proj_list = [] # 这个玩意儿是用来把backbone的输出投影到transformer输入的
        for _ in range(num_backbone_outs): # 将不同层（不同大小）的backbone的输出映射到相同大小
            in_channels = backbone.get_backbone()[1][_] # 这里面这个[0]是模型，[1]是输出尺寸列表
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
            )) # 投影
        for _ in range(num_feature_levels - num_backbone_outs): # 这循环在我们的参数下就运行一次
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(32, hidden_dim),
            )) # 把backbone的最后一层输出再做个卷积弄成更小的，论文里有张对应的图
            in_channels = hidden_dim
        self.input_proj = nn.ModuleList(input_proj_list)

        # 给上面那几个模块初始化，细节不太懂，反正是copy的
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = transformer.decoder.num_layers # 这玩意儿是最后预测使用几层decoder的feature_map，不使用两阶段方法就是这个数，否则这个数得+1表示在encoder输出层预测proposal

        # 最后预测时得用num_pred层的feature_map，所以把单层的embed堆叠起来得到多层的embed
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
        # 先拿backbone提个feature
        features = self.backbone(x)
        # 把不同尺寸的mask通过插值插出来
        masks = []
        for x in features:
            m = ms 
            assert m is not None
            masks.append(interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0])
        # 计算positional_encoding
        pos = []
        for l in range(len(features)):
            pos.append(self.position_encoding(features[l], masks[l]).to(features[l].dtype))
        # 把backbone的feature投影到transformer空间
        srcs = []
        for l in range(len(features)):
            srcs.append(self.input_proj[l](features[l]))
        # 投影多的那一层，backbone3层->transformer4层
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
        # 把上面这一大堆丢进transformer
        query_embeds = self.query_embed.weight
        hs, init_reference, inter_references = self.transformer(srcs, masks, pos, query_embeds)
        # hs是一个[6,4,300,256]的张量，表示6层，4个batch，300个object_queries的输出特征（256维）
        # 另外那俩是用来算reference_points的（因为box预测的是个偏移量，加上reference_points的坐标才是框）
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl]) # 预测类
            tmp = self.bbox_embed[lvl](hs[lvl]) # 预测偏移量
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid() # 预测坐标
            # 把每层预测的结果拼起来
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        

        if hasattr(self, 'Discriminator_1'):
            first_feature_map = features[0]
            first_GRL_features = grad_reverse(first_feature_map.clone())

            second_feature_map = features[1]
            second_GRL_features = grad_reverse(second_feature_map.clone())

            third_feature_map = features[2]
            third_GRL_features = grad_reverse(third_feature_map.clone())

            domain_predicts = []
            domain_predicts.append(nn.functional.softmax(self.domainClassifier(self.Discriminator_1(first_GRL_features)).clone(), dim=1))
            domain_predicts.append(nn.functional.softmax(self.domainClassifier(self.Discriminator_2(second_GRL_features)).clone(), dim=1))
            domain_predicts.append(nn.functional.softmax(self.domainClassifier(self.Discriminator_3(third_GRL_features)).clone(), dim=1))

            topk_hs = hs[-1, :, :10, :]
            topk_hs_GRL = grad_reverse(topk_hs.clone())
            for i in range(10): # set topk to be 10
                domain_predicts.append(nn.functional.softmax(self.domainFC(topk_hs_GRL[:, i, :]).clone(), dim=1))

            # 返回的时候把所有层和最后一层分别返回一下，虽然感觉很唐，但是为了能跟后面的接口接上还是这么写了
            out = {'logits_all': outputs_class, 'boxes_all': outputs_coord, 'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'domain_predict': domain_predicts}

            # print(f"outputs_coord shape is {outputs_coord.shape}, pred_boxes shape is {outputs_coord[-1].shape}")

        else:
            # 返回的时候把所有层和最后一层分别返回一下，虽然感觉很唐，但是为了能跟后面的接口接上还是这么写了
            out = {'logits_all': outputs_class, 'boxes_all': outputs_coord, 'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        return out


    def build_discriminators(self, device):

        self.Discriminator_1 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
        ).to(device)
        self.Discriminator_2 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
        ).to(device)
        self.Discriminator_3 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
        ).to(device)

        self.domainClassifier = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(False),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 2048),
            nn.ReLU(False),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2),
        ).to(device)

        self.domainFC = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(False),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(False),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 2),
        ).to(device)