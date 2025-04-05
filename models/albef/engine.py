import os
from typing import List
import torch
import re

from models.vlp_model import VLPModel
from .VL_Transformer_ITM import VL_Transformer_ITM, transform, getAttMap
from models.albef.models.tokenization_bert import BertTokenizer
import torch.nn.functional as F

from matplotlib import pyplot as plt
from PIL import Image
import cv2
import numpy as np
def mySig(x, a = 20, b = 0.5):
    sig_x = 1 / (1 + torch.exp(a*(-x+b)))
    sig_x = sig_x / sig_x.max()
    return sig_x
def process_gradcam(gradcam, threshold=0.5):
    B, H, W = gradcam.shape
    gradcam = gradcam.view(B, H*W)
    gradcam_stretched = mySig(gradcam)
    mask = (gradcam_stretched < threshold).long()
    return mask

class ALBEF(VLPModel):
    def __init__(self, model_id, device='cuda', templates = 'there is a {}', mode = 0, alpha = 0.2):
        super().__init__(model_id, device, templates=templates)
        
        self.model, self.tokenizer = self.load_model(model_id)
        self.gradcam = None
        self.gradcam_ori = None
        self.prev_itm_score = 0
        self.mode = mode
        self.alpha = alpha
    
    def load_model(self, model_id):
        if model_id is None:
            raise Exception("Model ID cannot be None.")

        if not self._models.has(model_id):
            tokenizer = BertTokenizer.from_pretrained('../checkpoints/bert-base-uncased')
            model = VL_Transformer_ITM(text_encoder='../checkpoints/bert-base-uncased',
                                       config_bert=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                'config_bert.json'))
            checkpoint_path = os.path.join(self.checkpoint_dir, f'{model_id}.pth')
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if "model" in checkpoint:
                checkpoint = checkpoint["model"]
            checkpoint = {k.replace('.bert', ''): v for k, v in checkpoint.items()}

            model.load_state_dict(checkpoint, strict=False)
            model.eval()
            model.to(self.device)
            self._models.put(model_id, (model, tokenizer))

        return self._models.get(model_id)

    
    
    def get_results_for_rec(
        self, 
        image_tensor: torch.tensor, 
        texts: List[str], 
        block_num: int = 8, 
        return_gradcams: bool = True
    ):
        
        self.model.text_encoder.encoder.layer[block_num].crossattention.self.save_attention = True
        texts_prompt = [self.templates.format(text) for text in texts]

        text_input = self.tokenizer(texts_prompt, return_tensors="pt", padding='max_length',
                               truncation=True, max_length=50).to(self.device)
        
        image_embeddings, vl_emveddings,output = self.model(image_tensor, text_input)
        loss = output[:, 1].sum()
        self.model.zero_grad()
        loss.backward()

        num_patch = int(self.model.visual_encoder.patch_embed.num_patches**0.5)
        

        with torch.no_grad():
            mask = text_input.attention_mask.view(text_input.attention_mask.size(0), 1, -1, 1, 1).cpu()

            cams = self.model.text_encoder.encoder.layer[block_num].crossattention.self.get_attention_map().cpu()
            grads = self.model.text_encoder.encoder.layer[block_num].crossattention.self.get_attn_gradients().cpu()

            cams = cams[:, :, :, 1:].reshape(image_tensor.size(0), 12, -1, num_patch, num_patch) * mask
            grads = grads[:, :, :, 1:].clamp(0).reshape(image_tensor.size(0), 12, -1, num_patch, num_patch) * mask

            gradcams = cams * grads
        
        start_idx = len(self.tokenizer(self.templates.format(""), return_tensors='pt').input_ids[0]) - 1
        
        all_gradcams = []
        
        for z, text in enumerate(texts):
            
            main_words_ids, id_map = self.find_main_words(text, start_idx, self.tokenizer)
            all_focus_ids = []

            for word_id in main_words_ids:
                all_focus_ids.extend(id_map[word_id])
            
            find_words = (len(all_focus_ids)>0)
            # Add [CLS]
            all_focus_ids.append(0)

            num_effective_text_token = text_input.attention_mask[z].count_nonzero().item()
            gradcam = gradcams[z]
            gradcam = gradcam.mean(0)[all_focus_ids, ...].mean(0) if find_words else \
                gradcam.mean(0)[:num_effective_text_token, ...].mean(0)
            gradcam = gradcam.view(1, 1, num_patch, num_patch)
            gradcam -= gradcam.min()
            if gradcam.max() > 0:
                gradcam /= gradcam.max()
            
            # gradcam = skimage_transform.resize(gradcam, (gradcam.shape), order = 3, mode = 'constant')

           
            all_gradcams.append(gradcam)
            
        final_gradcams = torch.cat(all_gradcams, dim=0).to(self.device)
        
        if return_gradcams:
            return final_gradcams
        
        return final_gradcams,image_embeddings,vl_emveddings
       
    def cal_score(
        self,
        gradcam,
        gt_bbox,
        boxes_category,
        od_scores,
        use_weighted_grade=True
    ):
        # cal box_output
        max_score = 0
        box_output = [0, 0, 0, 0]
        if not use_weighted_grade:
            od_scores = None
        for i, det in enumerate(boxes_category):
            score = gradcam[int(det[1]):int(det[3]), int(det[0]):int(det[2])]
            area = (det[3] - det[1]) * (det[2] - det[0])
            score = score.sum()
            score /= area ** 0.5
            
            if od_scores is not None:
                s = od_scores[i]
                coefficient = s
                score *= coefficient

            if score > max_score:
                max_score = score
                box_output = det[:4]
            
        x = 0
        if gt_bbox and self.cal_iou(box_output, gt_bbox) >= 0.5:
            x = 1
        return x, box_output
    
    def visualize_groundvlp(
        self,
        image_path,
        query,
        # texts,
        epoch = None,
        block_num=8,
        
    ):
        model, tokenizer = self.load_model(self.model_id)

        image_pil = Image.open(image_path).convert('RGB')
        image = transform(image_pil).unsqueeze(0).to(self.device)

        model.text_encoder.encoder.layer[block_num].crossattention.self.save_attention = True
        
        text_input = tokenizer(self.templates.format(query), return_tensors="pt").to(self.device)
        # texts_prompt = [self.templates.format(text) for text in texts]

        # text_input = tokenizer(texts_prompt, return_tensors="pt", padding='max_length',
        #                        truncation=True, max_length=50).to(self.device)
        if epoch is None or epoch == 0: output = model(image, text_input)
        else: output = model(image, text_input, self.gradcam)
        loss = output[:, 1].sum()
        itm_score = torch.nn.functional.softmax(output, dim=1)[:, 1].sum()
        model.zero_grad()
        loss.backward()

        num_patch = int(model.visual_encoder.patch_embed.num_patches**0.5)

        with torch.no_grad():
            mask = text_input.attention_mask.view(text_input.attention_mask.size(0), 1, -1, 1, 1).cpu()

            cams = model.text_encoder.encoder.layer[block_num].crossattention.self.get_attention_map().cpu()
            grads = model.text_encoder.encoder.layer[block_num].crossattention.self.get_attn_gradients().cpu()

            cams = cams[:, :, :, 1:].reshape(image.size(0), 12, -1, num_patch, num_patch) * mask
            grads = grads[:, :, :, 1:].clamp(0).reshape(image.size(0), 12, -1, num_patch, num_patch) * mask

            # gradcams = cams * grads
        
        # Find agent
        # agent = self.find_agent(query)
        # mapped_coco_label = self.map_to_coco_label(agent)
        # # Obtain the boxes_category
        # boxes_category, od_scores = self.get_bbox_for_rec(image_path, mapped_coco_label, threshold=0.15, general=True)

        # Visual-Word Attention Aggregation
        start_idx = len(tokenizer(self.templates.format(""), return_tensors='pt').input_ids[0]) - 1
        main_words_ids, id_map = self.find_main_words(query, start_idx, tokenizer)
        # main_words_ids, id_map = self.find_no_main_words(query, start_idx, tokenizer)
        main_word, main_word_id, length = self.find_agent(query)
        # print(main_word)
        all_focus_ids = []

        for word_id in main_words_ids:
            all_focus_ids.extend(id_map[word_id])
        alpha = self.alpha
        ############################################
        # No Main Word
        # num_effective_text_token = text_input.attention_mask[0].count_nonzero().item()
        # gradcam = gradcams[0]
        # gradcam = gradcam.mean(0)[:num_effective_text_token, ...].mean(0)
        # gradcam = gradcam.view(1, 1, num_patch, num_patch)
        # if epoch is not None:
        #     tmpgradcam = gradcam.clone().reshape(1, num_patch, num_patch)
        #     tmpgradcam -= tmpgradcam.min()
        #     if tmpgradcam.max() > 0:
        #         tmpgradcam /= tmpgradcam.max()
        #     if epoch == 0: 
        #         self.gradcam = process_gradcam(tmpgradcam)
        #         self.gradcam_ori = tmpgradcam
        #         gradcam = self.gradcam_ori.reshape(1, 1, num_patch, num_patch).to(gradcam.dtype)
        #         self.prev_itm_score = itm_score.item()
        #     else:
        #         now_gradcam = process_gradcam(tmpgradcam)
        #         self.gradcam = self.gradcam & now_gradcam
        #         alpha = 0.2 # * (itm_score.item()/self.prev_itm_score)
        #         self.gradcam_ori = self.gradcam_ori*(1-alpha) + tmpgradcam*alpha
        #         gradcam = self.gradcam_ori.reshape(1, 1, num_patch, num_patch).to(gradcam.dtype)
        #         self.prev_itm_score = itm_score.item()
        # gradcam = F.interpolate(gradcam, size=(image_pil.size[1], image_pil.size[0]),
        #                         mode='bicubic', align_corners=False).squeeze()
        # gradcam -= gradcam.min()
        # if gradcam.max() > 0:
        #     gradcam /= gradcam.max()
        ############################################
        if self.mode == 0:
            # CAMDiff Our's Enhancement
            if main_word_id != -1 : 
                len_now = len(all_focus_ids)
                while main_word_id not in id_map and main_word_id <= length+1: main_word_id += 1
                if main_word_id in id_map:
                    for _ in range(len_now):
                        all_focus_ids.extend(id_map[main_word_id])
                else:
                    raise ValueError("无法插入主词")
            
            if main_word_id == -1:
                find_words = (len(all_focus_ids)>0)
                # Add [CLS]
                all_focus_ids.append(0)

                num_effective_text_token = text_input.attention_mask[0].count_nonzero().item()
                # gradcam = gradcams[0]
                grad = grads[0]; cam = cams[0]
                # gradcam = gradcam.mean(0)[all_focus_ids, ...].mean(0) if find_words else \
                #     gradcam.mean(0)[:num_effective_text_token, ...].mean(0)
                if find_words:
                    gradcam = (grad.mean(0)[:num_effective_text_token, ...]) * (cam.mean(0)[:num_effective_text_token, ...])
                    gradcam = gradcam.mean(0)
                else:
                    gradcam = (grad.mean(0)[all_focus_ids, ...]) * (cam.mean(0)[all_focus_ids, ...])
                    gradcam = gradcam.mean(0)
                gradcam = gradcam.view(1, 1, num_patch, num_patch)
                if epoch is not None:
                    tmpgradcam = gradcam.clone().reshape(1, num_patch, num_patch)
                    tmpgradcam -= tmpgradcam.min()
                    if tmpgradcam.max() > 0:
                        tmpgradcam /= tmpgradcam.max()
                    if epoch == 0: 
                        self.gradcam = process_gradcam(tmpgradcam)
                        self.gradcam_ori = tmpgradcam
                        gradcam = self.gradcam_ori.reshape(1, 1, num_patch, num_patch).to(gradcam.dtype)
                        self.prev_itm_score = itm_score.item()
                    else:
                        now_gradcam = process_gradcam(tmpgradcam)
                        self.gradcam = self.gradcam & now_gradcam
                        # alpha = 0.2 # * (itm_score.item()/self.prev_itm_score)
                        self.gradcam_ori = self.gradcam_ori*(1-alpha) + tmpgradcam*alpha
                        gradcam = self.gradcam_ori.reshape(1, 1, num_patch, num_patch).to(gradcam.dtype) 
                        self.prev_itm_score = itm_score.item()
                gradcam = F.interpolate(gradcam, size=(image_pil.size[1], image_pil.size[0]),
                                        mode='bicubic', align_corners=False).squeeze()
                # print(gradcam)
                gradcam -= gradcam.min()
                if gradcam.max() > 0:
                    gradcam /= gradcam.max()
            else:
                id_nums = grads.shape[2]
                main_word_id = id_map[main_word_id]
                if len(main_word_id) == 1 : 
                    main_word_map_grads = grads[:, :, main_word_id, :, :]
                    main_word_map_cams  = cams[:, :, main_word_id, :, :]
                else: 
                    main_word_map_grads = grads[:, :, main_word_id, :, :].mean(2).unsqueeze(2)
                    main_word_map_cams  = cams[:, :, main_word_id, :, :].mean(2).unsqueeze(2)
                main_word_map = main_word_map_grads * main_word_map_cams
                all_focus_ids = all_focus_ids
                oth_focus_ids = set(all_focus_ids).copy()
                for id in main_word_id : oth_focus_ids.discard(id)
                oth_focus_ids = list(oth_focus_ids) # 去掉主词
                # oth_focus_ids = list(set(all_focus_ids)) # 仍然保留主词
                if len(oth_focus_ids)==0:
                    focus_words_map_grads = torch.zeros(grads.shape[0],grads.shape[1],1,grads.shape[3],grads.shape[4])
                    focus_words_map_cams =  torch.zeros(grads.shape[0],grads.shape[1],1,grads.shape[3],grads.shape[4])
                else:
                    focus_words_map_grads = grads[:,:,oth_focus_ids, :, :]
                    focus_words_map_cams =  cams[:,:,oth_focus_ids,:,:]
                
                
                
                # 计算L2范数
                # 在第三个维度上计算L2范数，需要保持其他维度不变
                main_minus_cam = (main_word_map_cams - focus_words_map_cams)
                norm = torch.norm(main_minus_cam, p=2, dim=2, keepdim=True)
                # max_val = torch.max(torch.abs(main_minus_cam), dim=2, keepdim=True)[0] ; 

                # 避免除以零，将零值替换为很小的数字
                norm = torch.where(norm == 0, torch.tensor(1e-8), norm)
                # max_val = torch.where(max_val == 0, torch.tensor(1e-8, dtype=max_val.dtype, device=max_val.device), max_val)

                # 进行归一化
                main_minus_cam = main_minus_cam / norm
                # main_minus_cam = (main_minus_cam) / max_val
                main_word_map_enc = main_minus_cam * main_word_map_grads * main_word_map
                # main_word_map_enc = torch.abs(main_word_map_cams - focus_words_map_cams).clamp(min=0, max=1.0) * main_word_map * main_word_map_grads # + main_word_map
                # main_word_map_enc = main_word_map - focus_words_map
                focus_words_map = grads[:,:,list(all_focus_ids), :, :] * cams[:,:,list(all_focus_ids),:,:]
                # 把增强主词+关注词+ID 拼接起来
                
                gradcams = torch.cat(((grads[:, :, 0, :, :]*cams[:, :, 0, :, :]).unsqueeze(2) ,main_word_map_enc, focus_words_map),dim=2).mean(2)
                
                gradcam = gradcams[0].mean(0)
                gradcam = gradcam.view(1, 1, num_patch, num_patch)
                if epoch is not None:
                    tmpgradcam = gradcam.clone().reshape(1, num_patch, num_patch)
                    tmpgradcam -= tmpgradcam.min()
                    if tmpgradcam.max() > 0:
                        tmpgradcam /= tmpgradcam.max()
                    if epoch == 0: 
                        self.gradcam = process_gradcam(tmpgradcam)
                        self.gradcam_ori = tmpgradcam
                        gradcam = self.gradcam_ori.reshape(1, 1, num_patch, num_patch).to(gradcam.dtype)
                        self.prev_itm_score = itm_score.item()
                    else:
                        now_gradcam = process_gradcam(tmpgradcam)
                        self.gradcam = self.gradcam & now_gradcam
                        # alpha = 0.2 # * (itm_score.item()/self.prev_itm_score)
                        self.gradcam_ori = self.gradcam_ori*(1-alpha) + tmpgradcam*alpha
                        gradcam = self.gradcam_ori.reshape(1, 1, num_patch, num_patch).to(gradcam.dtype)
                        self.prev_itm_score = itm_score.item()
                gradcam = F.interpolate(gradcam, size=(image_pil.size[1], image_pil.size[0]),
                                        mode='bicubic', align_corners=False).squeeze()
                # print(gradcam)
                gradcam -= gradcam.min()
                if gradcam.max() > 0:
                    gradcam /= gradcam.max()
        elif self.mode == 1:
            # No Selection
            gradcams = cams * grads
            num_effective_text_token = text_input.attention_mask[0].count_nonzero().item()
            gradcam = gradcams[0]
            gradcam = gradcam.mean(0)[:num_effective_text_token, ...].mean(0)
            gradcam = gradcam.view(1, 1, num_patch, num_patch)
            if epoch is not None:
                tmpgradcam = gradcam.clone().reshape(1, num_patch, num_patch)
                tmpgradcam -= tmpgradcam.min()
                if tmpgradcam.max() > 0:
                    tmpgradcam /= tmpgradcam.max()
                if epoch == 0: 
                    self.gradcam = process_gradcam(tmpgradcam)
                    self.gradcam_ori = tmpgradcam
                    gradcam = self.gradcam_ori.reshape(1, 1, num_patch, num_patch).to(gradcam.dtype)
                    self.prev_itm_score = itm_score.item()
                else:
                    now_gradcam = process_gradcam(tmpgradcam)
                    self.gradcam = self.gradcam & now_gradcam
                    alpha = 0.2 # * (itm_score.item()/self.prev_itm_score)
                    self.gradcam_ori = self.gradcam_ori*(1-alpha) + tmpgradcam*alpha
                    gradcam = self.gradcam_ori.reshape(1, 1, num_patch, num_patch).to(gradcam.dtype)
                    self.prev_itm_score = itm_score.item()
            gradcam = F.interpolate(gradcam, size=(image_pil.size[1], image_pil.size[0]),
                                    mode='bicubic', align_corners=False).squeeze()
            # print(gradcam)
            gradcam -= gradcam.min()
            if gradcam.max() > 0:
                gradcam /= gradcam.max()
        elif self.mode == 2:
            # GVLP
            gradcams = cams * grads
            find_words = (len(all_focus_ids)>0)
            # Add [CLS]
            all_focus_ids.append(0)
            num_effective_text_token = text_input.attention_mask[0].count_nonzero().item()
            gradcam = gradcams[0]
            gradcam = gradcam.mean(0)[all_focus_ids, ...].mean(0) if find_words else \
                gradcam.mean(0)[:num_effective_text_token, ...].mean(0)
            gradcam = gradcam.view(1, 1, num_patch, num_patch)
            if epoch is not None:
                tmpgradcam = gradcam.clone().reshape(1, num_patch, num_patch)
                tmpgradcam -= tmpgradcam.min()
                if tmpgradcam.max() > 0:
                    tmpgradcam /= tmpgradcam.max()
                if epoch == 0: 
                    self.gradcam = process_gradcam(tmpgradcam)
                    self.gradcam_ori = tmpgradcam
                    gradcam = self.gradcam_ori.reshape(1, 1, num_patch, num_patch).to(gradcam.dtype)
                    self.prev_itm_score = itm_score.item()
                else:
                    now_gradcam = process_gradcam(tmpgradcam)
                    self.gradcam = self.gradcam & now_gradcam
                    alpha = 0.2 # * (itm_score.item()/self.prev_itm_score)
                    self.gradcam_ori = self.gradcam_ori*(1-alpha) + tmpgradcam*alpha
                    gradcam = self.gradcam_ori.reshape(1, 1, num_patch, num_patch).to(gradcam.dtype)
                    self.prev_itm_score = itm_score.item()
            gradcam = F.interpolate(gradcam, size=(image_pil.size[1], image_pil.size[0]),
                                    mode='bicubic', align_corners=False).squeeze()
            # print(gradcam)
            gradcam -= gradcam.min()
            if gradcam.max() > 0:
                gradcam /= gradcam.max()
        
        elif self.mode == 3:
            # GVLP + n Main word
            gradcams = cams * grads            
            if main_word_id != -1 : 
                len_now = len(all_focus_ids)
                while main_word_id not in id_map and main_word_id <= length+1: main_word_id += 1
                if main_word_id in id_map:
                    for _ in range(len_now):
                        all_focus_ids.extend(id_map[main_word_id])
                else:
                    raise ValueError("无法插入主词")
            
            find_words = (len(all_focus_ids)>0)
            # Add [CLS]
            all_focus_ids.append(0)
            num_effective_text_token = text_input.attention_mask[0].count_nonzero().item()
            gradcam = gradcams[0]
            gradcam = gradcam.mean(0)[all_focus_ids, ...].mean(0) if find_words else \
                gradcam.mean(0)[:num_effective_text_token, ...].mean(0)
            gradcam = gradcam.view(1, 1, num_patch, num_patch)
            if epoch is not None:
                tmpgradcam = gradcam.clone().reshape(1, num_patch, num_patch)
                tmpgradcam -= tmpgradcam.min()
                if tmpgradcam.max() > 0:
                    tmpgradcam /= tmpgradcam.max()
                if epoch == 0: 
                    self.gradcam = process_gradcam(tmpgradcam)
                    self.gradcam_ori = tmpgradcam
                    gradcam = self.gradcam_ori.reshape(1, 1, num_patch, num_patch).to(gradcam.dtype)
                    self.prev_itm_score = itm_score.item()
                else:
                    now_gradcam = process_gradcam(tmpgradcam)
                    self.gradcam = self.gradcam & now_gradcam
                    alpha = 0.2 # * (itm_score.item()/self.prev_itm_score)
                    self.gradcam_ori = self.gradcam_ori*(1-alpha) + tmpgradcam*alpha
                    gradcam = self.gradcam_ori.reshape(1, 1, num_patch, num_patch).to(gradcam.dtype)
                    self.prev_itm_score = itm_score.item()
            gradcam = F.interpolate(gradcam, size=(image_pil.size[1], image_pil.size[0]),
                                    mode='bicubic', align_corners=False).squeeze()
            # print(gradcam)
            gradcam -= gradcam.min()
            if gradcam.max() > 0:
                gradcam /= gradcam.max()
        
        elif self.mode == 4:
            # Selectword + Diff + CLS
            if main_word_id == -1:
                find_words = (len(all_focus_ids)>0)
                # Add [CLS]
                all_focus_ids.append(0)

                num_effective_text_token = text_input.attention_mask[0].count_nonzero().item()
                # gradcam = gradcams[0]
                grad = grads[0]; cam = cams[0]
                # gradcam = gradcam.mean(0)[all_focus_ids, ...].mean(0) if find_words else \
                #     gradcam.mean(0)[:num_effective_text_token, ...].mean(0)
                if find_words:
                    gradcam = (grad.mean(0)[:num_effective_text_token, ...]) * (cam.mean(0)[:num_effective_text_token, ...])
                    gradcam = gradcam.mean(0)
                else:
                    gradcam = (grad.mean(0)[all_focus_ids, ...]) * (cam.mean(0)[all_focus_ids, ...])
                    gradcam = gradcam.mean(0)
                gradcam = gradcam.view(1, 1, num_patch, num_patch)
                if epoch is not None:
                    tmpgradcam = gradcam.clone().reshape(1, num_patch, num_patch)
                    tmpgradcam -= tmpgradcam.min()
                    if tmpgradcam.max() > 0:
                        tmpgradcam /= tmpgradcam.max()
                    if epoch == 0: 
                        self.gradcam = process_gradcam(tmpgradcam)
                        self.gradcam_ori = tmpgradcam
                        gradcam = self.gradcam_ori.reshape(1, 1, num_patch, num_patch).to(gradcam.dtype)
                        self.prev_itm_score = itm_score.item()
                    else:
                        now_gradcam = process_gradcam(tmpgradcam)
                        self.gradcam = self.gradcam & now_gradcam
                        # alpha = 0.2 # * (itm_score.item()/self.prev_itm_score)
                        self.gradcam_ori = self.gradcam_ori*(1-alpha) + tmpgradcam*alpha
                        gradcam = self.gradcam_ori.reshape(1, 1, num_patch, num_patch).to(gradcam.dtype) 
                        self.prev_itm_score = itm_score.item()
                gradcam = F.interpolate(gradcam, size=(image_pil.size[1], image_pil.size[0]),
                                        mode='bicubic', align_corners=False).squeeze()
                # print(gradcam)
                gradcam -= gradcam.min()
                if gradcam.max() > 0:
                    gradcam /= gradcam.max()
            else:
                id_nums = grads.shape[2]
                main_word_id = id_map[main_word_id]
                if len(main_word_id) == 1 : 
                    main_word_map_grads = grads[:, :, main_word_id, :, :]
                    main_word_map_cams  = cams[:, :, main_word_id, :, :]
                else: 
                    main_word_map_grads = grads[:, :, main_word_id, :, :].mean(2).unsqueeze(2)
                    main_word_map_cams  = cams[:, :, main_word_id, :, :].mean(2).unsqueeze(2)
                main_word_map = main_word_map_grads * main_word_map_cams
                all_focus_ids = all_focus_ids
                oth_focus_ids = set(all_focus_ids).copy()
                for id in main_word_id : oth_focus_ids.discard(id)
                oth_focus_ids = list(oth_focus_ids) # 去掉主词
                # oth_focus_ids = list(set(all_focus_ids)) # 仍然保留主词
                if len(oth_focus_ids)==0:
                    focus_words_map_grads = torch.zeros(grads.shape[0],grads.shape[1],1,grads.shape[3],grads.shape[4])
                    focus_words_map_cams =  torch.zeros(grads.shape[0],grads.shape[1],1,grads.shape[3],grads.shape[4])
                else:
                    focus_words_map_grads = grads[:,:,oth_focus_ids, :, :]
                    focus_words_map_cams =  cams[:,:,oth_focus_ids,:,:]
                
                
                
                # 计算L2范数
                # 在第三个维度上计算L2范数，需要保持其他维度不变
                main_minus_cam = (main_word_map_cams - focus_words_map_cams)
                norm = torch.norm(main_minus_cam, p=2, dim=2, keepdim=True)
                # max_val = torch.max(torch.abs(main_minus_cam), dim=2, keepdim=True)[0] ; 

                # 避免除以零，将零值替换为很小的数字
                norm = torch.where(norm == 0, torch.tensor(1e-8), norm)
                # max_val = torch.where(max_val == 0, torch.tensor(1e-8, dtype=max_val.dtype, device=max_val.device), max_val)

                # 进行归一化
                main_minus_cam = main_minus_cam / norm
                # main_minus_cam = (main_minus_cam) / max_val
                main_word_map_enc = main_minus_cam * main_word_map_grads * main_word_map
                # main_word_map_enc = torch.abs(main_word_map_cams - focus_words_map_cams).clamp(min=0, max=1.0) * main_word_map * main_word_map_grads # + main_word_map
                # main_word_map_enc = main_word_map - focus_words_map
                focus_words_map = grads[:,:,list(all_focus_ids), :, :] * cams[:,:,list(all_focus_ids),:,:]
                # 把增强主词+关注词+ID 拼接起来
                
                gradcams = torch.cat(((grads[:, :, 0, :, :]*cams[:, :, 0, :, :]).unsqueeze(2) ,main_word_map_enc, focus_words_map),dim=2).mean(2)
                
                gradcam = gradcams[0].mean(0)
                gradcam = gradcam.view(1, 1, num_patch, num_patch)
                if epoch is not None:
                    tmpgradcam = gradcam.clone().reshape(1, num_patch, num_patch)
                    tmpgradcam -= tmpgradcam.min()
                    if tmpgradcam.max() > 0:
                        tmpgradcam /= tmpgradcam.max()
                    if epoch == 0: 
                        self.gradcam = process_gradcam(tmpgradcam)
                        self.gradcam_ori = tmpgradcam
                        gradcam = self.gradcam_ori.reshape(1, 1, num_patch, num_patch).to(gradcam.dtype)
                        self.prev_itm_score = itm_score.item()
                    else:
                        now_gradcam = process_gradcam(tmpgradcam)
                        self.gradcam = self.gradcam & now_gradcam
                        # alpha = 0.2 # * (itm_score.item()/self.prev_itm_score)
                        self.gradcam_ori = self.gradcam_ori*(1-alpha) + tmpgradcam*alpha
                        gradcam = self.gradcam_ori.reshape(1, 1, num_patch, num_patch).to(gradcam.dtype)
                        self.prev_itm_score = itm_score.item()
                gradcam = F.interpolate(gradcam, size=(image_pil.size[1], image_pil.size[0]),
                                        mode='bicubic', align_corners=False).squeeze()
                # print(gradcam)
                gradcam -= gradcam.min()
                if gradcam.max() > 0:
                    gradcam /= gradcam.max()
        ############################################
        # Our‘s Enhancement
        # if main_word_id == -1:
        #     find_words = (len(all_focus_ids)>0)
        #     # Add [CLS]
        #     all_focus_ids.append(0)

        #     num_effective_text_token = text_input.attention_mask[0].count_nonzero().item()
        #     gradcam = gradcams[0]
        #     gradcam = gradcam.mean(0)[all_focus_ids, ...].mean(0) if find_words else \
        #         gradcam.mean(0)[:num_effective_text_token, ...].mean(0)
        #     gradcam = gradcam.view(1, 1, num_patch, num_patch)
        #     if epoch is not None:
        #         tmpgradcam = gradcam.clone().reshape(1, num_patch, num_patch)
        #         tmpgradcam -= tmpgradcam.min()
        #         if tmpgradcam.max() > 0:
        #             tmpgradcam /= tmpgradcam.max()
        #         if epoch == 0: 
        #             self.gradcam = process_gradcam(tmpgradcam)
        #             self.gradcam_ori = tmpgradcam
        #             gradcam = self.gradcam_ori.reshape(1, 1, num_patch, num_patch).to(gradcam.dtype)
        #             self.prev_itm_score = itm_score.item()
        #         else:
        #             now_gradcam = process_gradcam(tmpgradcam)
        #             self.gradcam = self.gradcam & now_gradcam
        #             alpha = 0.2 # * (itm_score.item()/self.prev_itm_score)
        #             self.gradcam_ori = self.gradcam_ori*(1-alpha) + tmpgradcam*alpha
        #             gradcam = self.gradcam_ori.reshape(1, 1, num_patch, num_patch).to(gradcam.dtype) 
        #             self.prev_itm_score = itm_score.item()
        #     gradcam = F.interpolate(gradcam, size=(image_pil.size[1], image_pil.size[0]),
        #                             mode='bicubic', align_corners=False).squeeze()
        #     # print(gradcam)
        #     gradcam -= gradcam.min()
        #     if gradcam.max() > 0:
        #         gradcam /= gradcam.max()
        # else:
        #     id_nums = gradcams.shape[2]
        #     main_word_id = id_map[main_word_id]
        #     if len(main_word_id) == 1 : main_word_map = gradcams[:, :, main_word_id, :, :]
        #     else: main_word_map = gradcams[:, :, main_word_id, :, :].mean(2).unsqueeze(2)
        #     all_focus_ids = set(all_focus_ids)
        #     oth_focus_ids = set(all_focus_ids).copy()
        #     for id in main_word_id : oth_focus_ids.discard(id)
        #     oth_focus_ids = list(oth_focus_ids) # 去掉主词
        #     # oth_focus_ids = list(set(all_focus_ids)) # 仍然保留主词
        #     if len(oth_focus_ids)==0:
        #         focus_words_map = torch.zeros(gradcams.shape[0],gradcams.shape[1],1,gradcams.shape[3],gradcams.shape[4])
        #     else:
        #         focus_words_map = gradcams[:,:,oth_focus_ids, :, :]
            
        #     main_word_map_enc = torch.abs(main_word_map - focus_words_map).clamp(min=0, max=1.0) * main_word_map + main_word_map
        #     # main_word_map_enc = main_word_map - focus_words_map
        #     focus_words_map = gradcams[:,:,list(all_focus_ids), :, :]
        #     # 把增强主词+关注词+ID 拼接起来
            
        #     gradcams = torch.cat((gradcams[:, :, 0, :, :].unsqueeze(2) ,main_word_map_enc, focus_words_map),dim=2).mean(2)
            
        #     gradcam = gradcams[0].mean(0)
        #     gradcam = gradcam.view(1, 1, num_patch, num_patch)
        #     if epoch is not None:
        #         tmpgradcam = gradcam.clone().reshape(1, num_patch, num_patch)
        #         tmpgradcam -= tmpgradcam.min()
        #         if tmpgradcam.max() > 0:
        #             tmpgradcam /= tmpgradcam.max()
        #         if epoch == 0: 
        #             self.gradcam = process_gradcam(tmpgradcam)
        #             self.gradcam_ori = tmpgradcam
        #             gradcam = self.gradcam_ori.reshape(1, 1, num_patch, num_patch).to(gradcam.dtype)
        #             self.prev_itm_score = itm_score.item()
        #         else:
        #             now_gradcam = process_gradcam(tmpgradcam)
        #             self.gradcam = self.gradcam & now_gradcam
        #             alpha = 0.2 # * (itm_score.item()/self.prev_itm_score)
        #             self.gradcam_ori = self.gradcam_ori*(1-alpha) + tmpgradcam*alpha
        #             gradcam = self.gradcam_ori.reshape(1, 1, num_patch, num_patch).to(gradcam.dtype)
        #             self.prev_itm_score = itm_score.item()
        #     gradcam = F.interpolate(gradcam, size=(image_pil.size[1], image_pil.size[0]),
        #                             mode='bicubic', align_corners=False).squeeze()
        #     # print(gradcam)
        #     gradcam -= gradcam.min()
        #     if gradcam.max() > 0:
        #         gradcam /= gradcam.max()
        
        ############################################
        # Add [CLS]
        # all_focus_ids.append(0)
        # id_nums = gradcams.shape[2]
        # id_vis = [False for _ in range(id_nums)]
        # for id in all_focus_ids: id_vis[id] = True 
        # unfocus_ids = [id for id in range(id_nums) if not id_vis[id]]
        # # Step 1: 根据id找到main_word_map，然后找到除了main_word之外所有剩下的context
        # main_word_map = gradcams[:, :, all_focus_ids, :, :]  # Shape: [B, num_heads, H/16, W/16]
        # context_map = gradcams[:,:,unfocus_ids,:,:]  # Shape: [B, num_heads, num_text_token-1, H/16, W/16]
        # # Step 2: 把剩下的context的所有map在text这个维度上进行求平均
        # main_word_map = main_word_map.mean(dim=2)
        # context_mean_map = context_map.mean(dim=2)  # Shape: [B, num_heads, H/16, W/16]
        # # Step 3: main_word_map和context_mean_map做相减，得到在main_word_map上比context多的部分
        # gradcams = main_word_map - context_mean_map  # Shape: [B, num_heads, H/16, W/16]
        # # step
        # gradcam = gradcams[0].mean(0)
        # gradcam = gradcam.view(1, 1, num_patch, num_patch)
        # gradcam = F.interpolate(gradcam, size=(image_pil.size[1], image_pil.size[0]),
        #                         mode='bicubic', align_corners=False).squeeze()
        # # print(gradcam)
        # gradcam -= gradcam.min()
        # if gradcam.max() > 0:
        #     gradcam /= gradcam.max()
        #############################################
        # Mainword Prev
        # find_words = (len(all_focus_ids)>0)
        # # Add [CLS]
        # all_focus_ids.append(0)
        # num_effective_text_token = text_input.attention_mask[0].count_nonzero().item()
        # gradcam = gradcams[0]
        # gradcam = gradcam.mean(0)[all_focus_ids, ...].mean(0) if find_words else \
        #     gradcam.mean(0)[:num_effective_text_token, ...].mean(0)
        # gradcam = gradcam.view(1, 1, num_patch, num_patch)
        # if epoch is not None:
        #     tmpgradcam = gradcam.clone().reshape(1, num_patch, num_patch)
        #     tmpgradcam -= tmpgradcam.min()
        #     if tmpgradcam.max() > 0:
        #         tmpgradcam /= tmpgradcam.max()
        #     if epoch == 0: 
        #         self.gradcam = process_gradcam(tmpgradcam)
        #         self.gradcam_ori = tmpgradcam
        #         gradcam = self.gradcam_ori.reshape(1, 1, num_patch, num_patch).to(gradcam.dtype)
        #         self.prev_itm_score = itm_score.item()
        #     else:
        #         now_gradcam = process_gradcam(tmpgradcam)
        #         self.gradcam = self.gradcam & now_gradcam
        #         alpha = 0.2 # * (itm_score.item()/self.prev_itm_score)
        #         self.gradcam_ori = self.gradcam_ori*(1-alpha) + tmpgradcam*alpha
        #         gradcam = self.gradcam_ori.reshape(1, 1, num_patch, num_patch).to(gradcam.dtype)
        #         self.prev_itm_score = itm_score.item()
        # gradcam = F.interpolate(gradcam, size=(image_pil.size[1], image_pil.size[0]),
        #                         mode='bicubic', align_corners=False).squeeze()
        # # print(gradcam)
        # gradcam -= gradcam.min()
        # if gradcam.max() > 0:
        #     gradcam /= gradcam.max()
        ###############################################
        
        
        self.show_groundvlp(
            image_path=image_path, 
            query=query, gradcam=gradcam, 
            category=None, 
            boxes_category=None, 
            od_scores=None
        )
        rgb_image = cv2.imread(image_path)[:, :, ::-1]
        gradcam_image = getAttMap(rgb_image, gradcam)
        np.clip(gradcam_image, 0., 1., out=gradcam_image)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # Original Image
        ax[0].imshow(rgb_image)
        ax[0].axis('off')
        ax[0].set_title('Original Image')

        # Grad-CAM Image
        ax[1].imshow(gradcam_image, cmap='jet')  # You can change the colormap as needed
        ax[1].axis('off')
        ax[1].set_title('Grad-CAM')
        # plt.show()
        # Save the visualization
        plt.savefig('gradcam_visualization.png')
        return gradcam, itm_score

        
    def show_groundvlp(
        self,
        image_path,
        query,
        gradcam,
        category,
        boxes_category,
        od_scores
    ):
        num_image = 2
        fig, ax = plt.subplots(num_image, 1, figsize=(15, 5 * num_image))

        bgr_image = cv2.imread(image_path)
        ax[0].imshow(bgr_image[:, :, ::-1])
        ax[0].set_yticks([])
        ax[0].set_xticks([])
        ax[0].set_xlabel(query, fontsize=15)

        rgb_image = cv2.imread(image_path)[:, :, ::-1]
        rgb_image = np.float32(rgb_image) / 255

        gradcam_image = getAttMap(rgb_image, gradcam)
        np.clip(gradcam_image, 0., 1., out=gradcam_image)
        # print(np.max(gradcam_image))
        ax[1].imshow(gradcam_image)
        ax[1].set_yticks([])
        ax[1].set_xticks([])
        ax[1].set_xlabel(query, fontsize=20)

        # # visualize detic bbox
        # cv2_img_2 = draw_boxes_cv2(image_path=image_path, box_list=boxes_category)
        # ax[2].imshow(cv2_img_2[:, :, ::-1])
        # ax[2].set_yticks([])
        # ax[2].set_xticks([])
        # ax[2].set_xlabel(f'category: {category}', fontsize=20)

        # _, box_output = self.cal_score(gradcam=gradcam, gt_bbox=None, boxes_category=boxes_category, od_scores=od_scores, use_weighted_grade=True)

        # cv2_img_3 = draw_boxes_cv2(image_path=image_path, box_list=[box_output])
        # ax[3].imshow(cv2_img_3[:, :, ::-1])
        # ax[3].set_yticks([])
        # ax[3].set_xticks([])
        # ax[3].set_xlabel('box_output', fontsize=20)

        save_dir = "./output"
        os.makedirs(save_dir, exist_ok=True)
        image_name = image_path.split('/')[-1]
        save_path = os.path.join(save_dir, image_name)
        plt.savefig(save_path)
        plt.close('all')

    
 