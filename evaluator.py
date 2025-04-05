import numpy as np
import torch

class Evaluator():
    def __init__(self):
        self.counters_by_iou = {iou: 0 for iou in [0.5, 0.6, 0.7, 0.8, 0.9]}
        self.total_intersection_area = 0
        self.total_union_area = 0
        self.ious_list = []
        pass
    def compute_mask_iou(self, outputs: torch.Tensor, labels: torch.Tensor, EPS=1e-6):
        assert outputs.shape[0] == 1; assert outputs.shape == labels.shape; assert len(outputs.shape) == 3
        outputs = outputs.int(); labels = labels.int()
        intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
        union = (outputs | labels).float().sum((1, 2))  # Will be zero if both are 0
        iou = (intersection + EPS) / (union + EPS)  # EPS is used to avoid division by zero
        
        iou, intersection, union = iou.item(), intersection.item(), union.item()
        
        for iou_threshold in self.counters_by_iou.keys():
            if iou > iou_threshold:
                self.counters_by_iou[iou_threshold] += 1
                
        self.total_intersection_area += intersection
        self.total_union_area += union
        self.ious_list.append(iou)
        
        return iou, intersection, union

    def evaluate(self):
        num_samples = len(self.ious_list)
        
        if num_samples == 0:
            print("No samples to evaluate.")
            return
        
        precision_at_k = np.array(list(self.counters_by_iou.values())) / num_samples
        overall_iou = self.total_intersection_area / self.total_union_area
        mean_iou = np.mean(self.ious_list)

        print("Evaluation Result")
        iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        for iou, prec in zip(iou_thresholds, precision_at_k):
            print(f"IoU: {iou:.2f}, Precision: {prec:.4f}")
        print("========================================")
        print(f"Overall IoU: {overall_iou:.4f}")
        print(f"Mean IoU: {mean_iou:.4f}")