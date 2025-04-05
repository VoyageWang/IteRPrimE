import torch
import scipy.ndimage

def mySig(x, a = 20, b = 0.5):
    sig_x = 1 / (1 + torch.exp(a*(-x+b)))
    sig_x = sig_x / sig_x.max()
    return x

def post_process(bin_mask, gradcam, caption, select_idx, alpha = 1.00):
    """
    后处理操作
    
    Args:
        bin_mask (torch.Tensor): 候选mask tensor, shape=(1, 100, H, W)
        gradcam (torch.Tensor): GradCAM注意力图tensor, shape=(H, W)
        
    Returns:
        torch.Tensor: 最终mask, shape=(1, 1, H, W)
    """
    B, N, H, W = bin_mask.size()
    
    device = bin_mask.device
    max_val = torch.max(gradcam)
    max_coord = torch.nonzero(gradcam >= max_val * alpha)
    # 先去做一个判断初筛，看最大的这个点的位置是否是白色像素，这样就能够有效的取出
    bin_mask_at_seed = torch.zeros((B,N), dtype=torch.bool).to(device)
    for coord in max_coord:
        max_x, max_y = coord[0].item(), coord[1].item()
        bin_mask_at_seed_tmp = bin_mask[:, :, max_x, max_y]
        bin_mask_at_seed = torch.logical_or(bin_mask_at_seed, bin_mask_at_seed_tmp != 0.0)
    
    # 检查每个候选mask是否在seed位置有白色像素
    
    # 初筛2
    bin_mask_tmp = bin_mask.squeeze(0)
    res = []
    for mask in bin_mask_tmp:
        mask_np = mask.cpu().numpy()
        labeled_array, num_features = scipy.ndimage.label(mask_np)
        res.append(num_features<=12)
    res = torch.tensor(res).float().unsqueeze(0).to(device)
    bin_mask_at_seed = torch.logical_and(bin_mask_at_seed, res)
    
    bin_mask_at_seed = bin_mask_at_seed.float()
    # 对于初筛1，如果没有，将对应的mask置零, 然后后面进行初筛2
    bin_mask_tmp = bin_mask * bin_mask_at_seed.unsqueeze(-1).unsqueeze(-1)
    
    
    # gradcam = gradcam_reweight(caption, gradcam)
    
    # gradcam = torch.exp(gradcam)
    # 1. 计算每个候选mask和gradcam的点乘分数
    # nw = 1; gradcam = torch.exp(gradcam * nw) / torch.exp(torch.tensor(nw)) # 对gradcam重新加权一下
    # thresh = 0.5; 
    # gradcam = (gradcam >= thresh).float() 
    # gradcam = mySig(gradcam)
    scores = (bin_mask_tmp + bin_mask_tmp*gradcam.unsqueeze(0).unsqueeze(0)).sum((2, 3))  # (1, 100)
    
    w_area = bin_mask_tmp.sum((2,3)) + 1e-5
    
    scores /= w_area
    
    # 2. 选择分数最高的候选mask
    sorted_tensor, indices = torch.sort(scores,descending=True)
    for indice in indices[0]:
        if indice.item() in select_idx: continue
        max_idx = indice.item()
        break
    # max_score, max_idx = scores.view(200).max(0)
    print(max_idx)
    candidate_mask = bin_mask_tmp[0, max_idx]  
    
    # 3. 找到分数最高点作为种子点, 进行区域提取
    
    
    # seed_point = (max_x, max_y)
    # final_mask = extract_region(candidate_mask, seed_point).unsqueeze(0).to(device)
    
    return candidate_mask, max_idx

if __name__ == "__main__":
    bin_mask = torch.randn(1, 200, 480, 640)
    gradcam = torch.randn(480, 640)
    select_idx = []
    caption = ''
    
    