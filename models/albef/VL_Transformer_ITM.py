from functools import partial

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import filters

from models.albef.models.tokenization_bert import BertTokenizer
from models.albef.models.vit import VisionTransformer
from models.albef.models.xbert import BertConfig, BertModel
from skimage import transform as skimage_transform

import torch
from torch import nn
from torchvision import transforms
import re
from PIL import Image

# coding=utf-8
# 构建SPP层(空间金字塔池化层)
import math
import torch
import torch.nn.functional as F

# 构建SPP层(空间金字塔池化层)
class SPPLayer(torch.nn.Module):

    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        num, n,c = x.size()  # num:样本数量 c:通道数 h:高 w:宽

        h = 16
        w = 16
        # 提取最初的《cls》token
        cls_token = x[:, 0, :].view(num, 1, c)

        # 对剩下的 image token 做 SPP
        x = x[:, 1:, :].view(num, c, h, w)
        original_num_tokens = 256

        for i in range(self.num_levels):
            level = i + 1
            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            stride = (math.ceil(h / level), math.ceil(w / level))
            pooling = (math.floor((kernel_size[0]*level-h+1)/2), math.floor((kernel_size[1]*level-w+1)/2))

            # 选择池化方式
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling)

            # 展开、拼接
            if (i == 0):
                x_flatten = tensor.view(num, -1, c)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1, c)), 1)

       # 将最初的《cls》token拼接回去
        # x_flatten = torch.cat((cls_token, x_flatten), 1)

        # 复制 cls token 到 257 个
        cls_tokens = cls_token.expand(-1, original_num_tokens-x_flatten.shape[1]+2, -1)
        x_flatten = torch.cat((cls_tokens, x_flatten[:, 1:, :]), 1)


        return x_flatten


class VL_Transformer_ITM(nn.Module):
    def __init__(self,
                 text_encoder=None,
                 config_bert=''
                 ):
        super().__init__()

        bert_config = BertConfig.from_json_file(config_bert)

        self.visual_encoder = VisionTransformer(
            img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)

        self.itm_head = nn.Linear(768, 2)
        
        self.spp = SPPLayer(6)

    def forward(self, image, text, gradcam=None):
        image_embeds = self.visual_encoder(image)
        # 看看这里如果加了细粒度的特征会怎么样
        # image_embeds = image_embeds.reshape(1,-1,16,16)
        # image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        # image_embeds = self.spp(image_embeds)
        
        if gradcam is None : image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        else: 
            B = image_embeds.shape[0]
            ones = torch.ones(B, 1, dtype=torch.long, device=image.device)
            image_atts = torch.cat((ones, gradcam.to(image.device)), dim=1)
        
        output = self.text_encoder(text.input_ids.to(image.device),
                                   attention_mask=text.attention_mask.to(image.device),
                                   encoder_hidden_states=image_embeds,
                                   encoder_attention_mask=image_atts.to(image.device),
                                   return_dict=True,
                                   )
        vl_embeddings = output.last_hidden_state[:, 0, :]
        
        # print(vl_embeddings)
        vl_output = self.itm_head(vl_embeddings)
        # _,pred = torch.max(vl_output,1)
        # print(type(vl_output))
        # return image_embeds, output.last_hidden_state, vl_output
        return vl_output
    


def getAttMap(img, attMap, blur = True, overlap = True):
    attMap -= attMap.min()
    if attMap.max() > 0:
        attMap /= attMap.max()
    attMap = skimage_transform.resize(attMap, (img.shape[:2]), order = 3, mode = 'constant')
    # attMap = skimage_transform.resize(attMap, (attMap.shape), order = 3, mode = 'constant')
    
    if blur:
        attMap = filters.gaussian_filter(attMap, 0.02*max(img.shape[:2]))
        attMap -= attMap.min()
        attMap /= attMap.max()
    cmap = plt.get_cmap('jet')
    attMapV = cmap(attMap)
    attMapV = np.delete(attMapV, 3, 2)
    if overlap:
        attMap = 1*(1-attMap**0.7).reshape(attMap.shape + (1,))*img + (attMap**0.7).reshape(attMap.shape+(1,)) * attMapV
    return attMap

normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    normalize,
])




