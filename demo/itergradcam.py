import os
import shutil
import cv2 
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform as skimage_transform
from scipy.ndimage import filters
from PIL import Image
import cv2

def mySig(x, a = 20, b = 0.5):
    sig_x = 1 / (1 + torch.exp(a*(-x+b)))
    sig_x = sig_x / sig_x.max()
    return sig_x

def getAttMap(img, attMap, blur = False, overlap = True):
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
    
def show_groundvlp(image_path,query,gradcam,save_path = None):
    # category,
    # boxes_category,
    # od_scores
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
    # plt.show()
    if save_path:
        plt.savefig(save_path)


def IterGradCAM(engine,img_path,query,iter_num,path = './erase_imgs',vis = False,lmbda = 0.8):
    # Iterative机制
    gradcam = None
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    
    itm_scores = []
    pre_gradcam = None 
    flag = True
    for i in range(iter_num):
        # gradcam, itm_score = engine.visualize_groundvlp(image_path=img_path,query=query,epoch=i)
        if i != 0:
            gradcam, itm_score = engine.visualize_groundvlp(image_path=img_path,query=query,epoch=i)
            # tmp = torch.exp(tmp) / torch.exp(torch.tensor(1.0))
            # tmp = mySig(tmp)
            if grad_rev_sum.item()*itm_score.item() < itm_scores[-1]: 
            # if itm_score.item() < itm_scores[-1]: 
                flag = False
                break
            # gradcam = gradcam*lmbda + tmp*(1-lmbda)
            # gradcam = gradcam*(1-lmbda) + tmp*lmbda
            # gradcam = gradcam + tmp
            # gradcam = gradcam.clamp(0.0, 1.0)
            # gradcam -= gradcam.min()
            # if gradcam.max() > 0:
            #     gradcam /= gradcam.max()
        else:
            gradcam, itm_score = engine.visualize_groundvlp(image_path=img_path,query=query,epoch=i)
            # gradcam = torch.exp(gradcam) / torch.exp(torch.tensor(1.0))
            gradcam = mySig(gradcam)
            Image.open(img_path).save(f"{path}/raw.jpg")
            # img_path = f"{path}/raw.jpg"
            pre_gradcam = torch.zeros_like(gradcam)
        

        grad_rev_sum = torch.sum(torch.ones_like(pre_gradcam)-pre_gradcam) / pre_gradcam.numel()
        
        itm_scores.append(itm_score.item() * grad_rev_sum.item())
        print(f"Iteration {i}")
        
        pre_gradcam = gradcam
        
        if vis: show_groundvlp(f"{path}/raw.jpg",query,gradcam,f"{path}/{i}_grad.jpg")
    # show_groundvlp(f"{path}/raw.jpg",query,gradcam,f"{path}/result.jpg")
    if flag: i = iter_num
    return gradcam, i
def seed_torch(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True 

if __name__ == "__main__":
    # 初始化路径和query
    import sys
    sys.path.insert(1, os.path.join(sys.path[0], '..'))
    from models.albef.engine import ALBEF
    seed = 1234
    seed_torch(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    img_path = "/home/vegetabot/Filesys/CodeField_win/referformer_modify/data/coco/train2014/COCO_train2014_000000274266.jpg"
    query = "a man standing next to a young girl on a grassy hillside"  # "a lady pouring wine in a glass"
    
    

    engine = ALBEF(model_id='ALBEF', device='cuda', templates='there is a {}')
    IterGradCAM(engine,img_path,query,3, vis = True)