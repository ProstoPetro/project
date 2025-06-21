from scipy.optimize import linear_sum_assignment
import torch
from torchmetrics import Metric
from torchvision.ops import box_iou



class BoxRecall(Metric):
    """
    Boxes should be in xyxy format
    """
    def __init__(self, threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.add_state('true_positives', default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx='sum')
        self.add_state('positives', default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx='sum')
        self.threshold=threshold

    def update(self, preds, targets, image_indices):
        device = preds.device
        self.positives = self.positives.to(device)
        self.true_positives = self.true_positives.to(device)
        self.positives += targets.shape[0]
        for batch_idx in range(preds.shape[0]):
            mask = (image_indices == batch_idx)
            if torch.any(mask):
                self.true_positives += (count_matched_boxes(preds[batch_idx], targets[mask], self.threshold)).to(device)
    
    def compute(self):
        if self.positives == 0:
            return torch.tensor(torch.nan).to(self.positives.device)
        return self.true_positives / self.positives


def count_matched_boxes(pred_boxes, target_boxes, iou_th=0.5):
    iou_binary_matrix = box_iou(pred_boxes, target_boxes) >= iou_th
    return iou_binary_matrix.max(dim=0).values.sum()


class GroundingMetrics(Metric):
    """
    Computes a dict of accuracy for all iou `thresholds` and mean IoU
    Boxes should be in xyxy format
    """
    def __init__(self, thresholds, **kwargs):
        super().__init__(**kwargs)
        self.add_state('ious', default=[], dist_reduce_fx='cat')
        self.add_state('false_negatives', default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx='sum')
        self.add_state('false_positives', default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx='sum')
        self.thresholds=thresholds

    def update(self, preds, targets):
        """
        Inputs are boxes ((N, 4) and (M, 4)) for one image for one prompt
        """
        device = preds.device
        self.false_negatives.to(device)
        self.false_positives.to(device)
        if targets.shape[0] > preds.shape[0]:
            self.false_negatives += targets.shape[0] - preds.shape[0]
        if preds.shape[0] > targets.shape[0]:
            self.false_positives += preds.shape[0] - targets.shape[0]
        if targets.shape[0] > 0 and preds.shape[0] > 0:
            iou_matrix = box_iou(preds, targets).cpu()
            pred_indices, target_indices = linear_sum_assignment(-iou_matrix)
            self.ious.extend(iou_matrix[pred_indices, target_indices])

    def compute(self):
        if not len(self.ious):
            return torch.nan
        ious = torch.tensor(self.ious)
        metrics_dict = {'MeanIoU': torch.mean(ious[ious >= 0])}
        for th in self.thresholds:
            metrics_dict[f'Accuracy{th}'] = torch.sum(ious > th) / (len(ious) + self.false_positives + self.false_negatives)
            metrics_dict[f'Recall{th}'] = torch.sum(ious > th) / (len(ious) + self.false_negatives)
            metrics_dict[f'Precision{th}'] = torch.sum(ious > th) / (len(ious) + self.false_positives)
        return metrics_dict