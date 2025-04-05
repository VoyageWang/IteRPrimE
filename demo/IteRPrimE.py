# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os
import math
from Logger import Logger
from postproc import post_process
# from itergradcam import IterGradCAM
from itergradcam import IterGradCAM
# from sam_utils import SAM_MaskGen

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings
import torch
import cv2
import numpy as np
import tqdm
from PIL import Image

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo

from models.albef.engine import ALBEF
from models.albef.VL_Transformer_ITM import VL_Transformer_ITM
from models.albef.models.tokenization_bert import BertTokenizer

from collections import deque
from torchvision import transforms
import torch
from collections import deque



import torch
import argparse
from torch.utils.data import DataLoader, DistributedSampler
from dataset import build_dataset
import dataset.samplers as samplers
from evaluator import Evaluator
import util.misc as utils
import matplotlib.pyplot as plt

# class Args(argparse.Namespace):
#      masks = True
#      coco_path = "../data/coco"
#      distributed = False
#      num_workers = 4
     
def Args():
    parser = argparse.ArgumentParser(description="IteRPrimE demo for builtin configs")
    parser.add_argument("--masks", type=bool, default=True, help="Whether to use masks (default: True)")
    parser.add_argument("--coco-path", type=str, default="../data/coco", help="Path to the COCO dataset (default: ../data/coco)")
    parser.add_argument("--distributed", type=bool, default=False, help="Whether to use distributed training (default: False)")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading (default: 4)")
    
    parser.add_argument("--exp-name", type=str, default="IteRPrimE", help="Experiment name (default: IterVLP_OURS_CAM_MINUS_SOTA3)")
    parser.add_argument("--data-set", type=str, default="refcoco", help="Dataset name (default: refcoco)")
    parser.add_argument("--image-set", type=str, default="testA", help="Image set name (default: testA)")
    parser.add_argument("--iter-num", type=int, default=3, help="Number of iterations (default: 3)")
    parser.add_argument("--mode", type=int, default=0, help="Mode (default: 0)")
    parser.add_argument("--alpha", type=float, default=0.2, help="Alpha value (default: 0.2)")
    
    return parser
    

def extract_region(image, seed_point):
    # 复制原始图像,以免修改原始数据
    # output = image.clone()
    
    # 确保图像和掩码位于同一设备上
    device = image.device
    
    # 创建一个掩码来存储结果
    mask = torch.zeros_like(image, device=device)
    
    # 使用广度优先搜索算法
    queue = deque([(seed_point[0], seed_point[1])])
    target_value = image[seed_point[0], seed_point[1]]
    
    while queue:
        x, y = queue.popleft()
        
        if 0 <= x < image.shape[0] and 0 <= y < image.shape[1] and image[x, y] == target_value and mask[x, y] == 0:
            mask[x, y] = 1
            queue.append((x + 1, y))
            queue.append((x - 1, y))
            queue.append((x, y + 1))
            queue.append((x, y - 1))
    
    # 返回结果掩码
    return mask

def left_weight(h, w, start = 1.5, end = 0.5):
    sequence = torch.linspace(start, end, w)
    sequence = sequence.unsqueeze(0).expand(h, -1)
    return sequence

def right_weight(h, w, start = 0.5, end = 1.5):
    sequence = torch.linspace(start, end, w)
    sequence = sequence.unsqueeze(0).expand(h, -1)
    return sequence

def up_weight(h, w, start = 1.5, end = 0.5):
    sequence = torch.linspace(start, end, h)
    sequence = sequence.unsqueeze(1).expand(-1, w)
    return sequence

def down_weight(h, w, start = 0.5, end = 1.5):
    sequence = torch.linspace(start, end, h)
    sequence = sequence.unsqueeze(1).expand(-1, w)
    return sequence

def gradcam_reweight(caption, gradcam_weight):
    h, w = gradcam_weight.shape
    device = gradcam_weight.device
    if "left"  in caption: gradcam_weight = left_weight(h, w).to(device) * gradcam_weight
    if "right" in caption: gradcam_weight = right_weight(h, w).to(device) * gradcam_weight
    if "up"    in caption or "top" in caption: gradcam_weight = up_weight(h, w).to(device) * gradcam_weight
    if "down"  in caption or "bottom" in caption: gradcam_weight = down_weight(h, w).to(device) * gradcam_weight
    return gradcam_weight

def vis_mask(bin_mask,save_path="mask1/"):
    to_pil = transforms.ToPILImage()
    # 遍历每个掩码
    for i, mask in enumerate(bin_mask.float()):
        # 将二值掩码乘以255，将其转换为灰度图像
        mask.unsqueeze(0)
        pil_image = to_pil(mask)
        # image_path = f"mask1/mask_{i}.png"
        # pil_image.save(image_path)
        #将图像保存为PNG格式
        file_path = f'{save_path}/mask_{i}.png'
        pil_image.save(file_path)
        # cv2.imwrite(file_path, mask)

        print(f'Mask {i} saved to {file_path}')
def vis_singl_mask(bin_mask,save_path="mask1/",name='1'):
    bin_mask = bin_mask.cpu()
    to_pil = transforms.ToPILImage()
    pil_image = to_pil(bin_mask)
    file_path = f'{save_path}/{name}.png'
    pil_image.save(file_path)

# constants
WINDOW_NAME = "IteRPrimE demo"

def vis_result(image_path, mask_tensor):
    # 将mask张量转换为numpy数组并拉伸到3个通道
    image = Image.open(image_path).convert('RGBA')

    mask_np = mask_tensor.squeeze(0).cpu().numpy()
    # mask_np = np.stack([mask_np] * 3, axis=-1)  # (H, W, 3)

    # 将mask值为1的区域设置为红色，并且为半透明
    red_color = np.array([255, 0, 0, 128], dtype=np.uint8)  # 半透明红色
    red_mask = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    red_mask[mask_np.astype(bool)] = red_color

    # 将原始图像转换为带有alpha通道的图像
    image_np = np.array(image)

    # 将红色遮罩叠加到原始图像上
    result_np = image_np.copy()
    alpha_mask = red_mask[:, :, 3] / 255.0
    for c in range(3):
        result_np[:, :, c] = (1.0 - alpha_mask) * result_np[:, :, c] + alpha_mask * red_mask[:, :, c]

    result_image = Image.fromarray(result_np.astype(np.uint8))
    return result_image

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

class get_parser:
    config_file = "../configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml"
    opts = []

def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False

def iter_select(demo, engine, path, caption, iter_num, sam = None):
    def judge(string):
        string = string.lower()
        if "left" in string: return True 
        if "right" in string: return True 
        if "up" in string: return True 
        if "down" in string: return True 
        if "top" in string: return True 
        if "bottom" in string: return True 
        return False
    mask_iter_path = './mask_iter_img/'
    os.makedirs(os.path.dirname(mask_iter_path), exist_ok=True)
    print(f'Now check : {path} | Caption : {caption}')
    
    
    if not sam :
        img = read_image(path, format="BGR") 
        predictions, visualized_output, bin_mask = demo.run_on_image(img)
    else:
        print('SAM segmentation') 
        bin_mask = sam.run(path)
    bin_mask = bin_mask.unsqueeze(0)
    select_idx = []
    area = bin_mask.shape[-1] * bin_mask.shape[-2]
    cnt = 0
    itm_scores = []
    results = []
    Image.open(path).save(f"{mask_iter_path}{cnt}.jpg")
    # if judge(caption): thresh = 20
    # else: thresh = 1
    thresh = 1
    
    while cnt < thresh: # The Max Iter is 20
        cnt += 1
        # img = read_image(path, format="BGR")
        # predictions, visualized_output, bin_mask = demo.run_on_image(img)
        # bin_mask = bin_mask.unsqueeze(0)
        
        # gradcam, itm_score = engine.visualize_groundvlp(image_path=path,query=caption)
        gradcam, nums = IterGradCAM(engine, path, caption, iter_num, vis = True)
        gradcam = gradcam.to(bin_mask.device)
        
        result, idx = post_process(bin_mask, gradcam, caption, select_idx)
        
        area -= (bin_mask[0, idx].sum()).item()
        
        # if cnt > 1 and itm_score.item()*area*math.exp(1-(cnt/20)) < itm_scores[-1]: 
        #     print("=======================")
        #     return results[-1], select_idx
        
        # itm_scores.append(itm_score.item()*area*math.exp(1-(cnt/20)))
        
        results.append(result)
        select_idx.append(idx)
        # mask the image
        image_tensor = torch.tensor(np.array(Image.open(path).convert('RGB'))).permute(2, 0, 1).float()
        rand_mask = torch.rand(image_tensor.shape) * 255.0
        # image_tensor = (1-result.to(image_tensor.device)) * image_tensor + result.to(image_tensor.device) * rand_mask
        image_tensor = image_tensor + result.to(image_tensor.device) * rand_mask
        # image_tensor = (1-result.to(image_tensor.device)) * image_tensor
        result_image = Image.fromarray(image_tensor.permute(1, 2, 0).byte().numpy())
        result_image.save(f"{mask_iter_path}{cnt}.jpg")
        path = f"{mask_iter_path}{cnt}.jpg"
    print("=======================")
    return results[-1], select_idx, nums

import os
import random
import numpy as np
import torch

def seed_torch(seed):
    """
    Set the random seed for various modules to ensure reproducibility.
    
    Args:
    seed (int): The seed value to be set.
    """
    # Set PYTHONHASHSEED environment variable
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set random seed for Python's built-in random module
    random.seed(seed)
    
    # Set random seed for numpy
    np.random.seed(seed)
    
    # Set random seed for PyTorch
    torch.manual_seed(seed)
    
    # Set random seed for PyTorch (CUDA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    
    # Ensure that the cuDNN library's benchmark mode is disabled, and that 
    # cuDNN is deterministic.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    seed_torch(1234)
    args1 = Args().parse_args()
    exp_name, data_set, image_set = args1.exp_name, args1.data_set, args1.image_set
    Iter_num, mode, alpha = args1.iter_num, args1.mode, args1.alpha
    
    text_name = exp_name + '_' + data_set + '_' + image_set + f"_Iter{Iter_num}" + f"_mode{mode}" + f"_alpha_{alpha}" 
    txt_path = f"./logs/{text_name}.txt"
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    if os.path.exists(txt_path):
        os.remove(txt_path)
    dataset_val = build_dataset(data_set, image_set=image_set, args=args1)
    sampler_val = (
        samplers.DistributedSampler(dataset_val, shuffle=False) if args1.distributed else torch.utils.data.SequentialSampler(dataset_val)
    )
    data_loader_val = DataLoader(
            dataset_val,
            1, # args.batch_size,
            sampler=sampler_val,
            drop_last=False,
            collate_fn=utils.collate_fn,
            num_workers=args1.num_workers,
        )

    eval_agent = Evaluator()
        
    mp.set_start_method("spawn", force=True)
    args = get_parser()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    engine = ALBEF(model_id='ALBEF', device='cuda', templates='there is a {}', mode = mode, alpha = alpha)
    sam = None

    
    for i, (samples, targets) in enumerate(tqdm.tqdm(data_loader_val)):
        seed_torch(1234)
        # 数据打包
        captions = [t["caption"] for t in targets]
        realmask = [t['masks'] for t in targets];  realmask = torch.cat(realmask, dim=0)
        # realmask = realmask.squezze
        paths = [t["path"] for t in targets]
        path = paths[0]
        result, select_idx, nums = iter_select(demo, engine, path, captions[0], Iter_num, sam)
        # use PIL, to be consistent with evaluation
        
        
        result = result.unsqueeze(0)
        
        iou, intersection, union = eval_agent.compute_mask_iou(result.to("cpu"), realmask)
        
        output_string = path + '   iou: {}   '.format(iou) + "  I: {}".format(intersection) + "  U: {}  ".format(union)  + "\n"
        
        
        with open(txt_path, 'a') as file:
            if i == 0: file.write("-----------------START-----------------\n")
            file.write('No. '+ str(i) +' || ' + captions[0]+'   ||   nums: '+str(nums)+'   ||   '+' '.join(str(item) for item in select_idx) + '\n')
            file.write(output_string)

    sys.stdout = Logger(txt_path, mode='a')
    eval_agent.evaluate()
        
    
