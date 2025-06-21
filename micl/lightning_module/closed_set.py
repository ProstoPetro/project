import lightning as pl
import torch
from scipy.optimize import linear_sum_assignment
from torch import optim
from torch.nn import functional as F
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import generalized_box_iou_loss

from ..losses import generalized_cross_entropy, masked_background_bce_loss
from ..matcher import hungarian_matching


class DetectionModule(pl.LightningModule):
    def __init__(
        self,
        model,
        num_classes,
        clf_coeff=1.0,
        giou_coeff=2.0,
        l1_coeff=5.0,
        gce_cost=2.0,
        giou_cost=2.0,
        l1_cost=5.0,
        lr=1e-4,
        weight_decay=1e-4
    ):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.clf_coeff = clf_coeff
        self.giou_coeff = giou_coeff
        self.l1_coeff = l1_coeff
        self.gce_cost = gce_cost
        self.giou_cost = giou_cost
        self.l1_cost = l1_cost
        self.lr = lr
        self.weight_decay = weight_decay
        self.map_metric = MeanAveragePrecision()


    def compute_loss(self, pred_boxes, pred_logits, target_boxes, target_labels):
        indices = hungarian_matching(pred_boxes, target_boxes, pred_logits, target_labels,
                                     gce_cost=self.gce_cost, giou_cost=self.giou_cost, l1_cost=self.l1_cost)
        
        device = pred_logits.device
        matched_pred_boxes = select_and_pad(pred_boxes, target_boxes, indices)
        matched_pred_logits = select_and_pad(pred_logits, target_labels, indices)
        
        if torch.any(matched_pred_boxes != -1):
            gce_loss = generalized_cross_entropy(matched_pred_logits, target_labels)
            giou_loss = generalized_box_iou_loss(matched_pred_boxes, target_boxes)
            l1_loss = F.l1_loss(matched_pred_boxes, target_boxes)
        else:
            gce_loss = torch.tensor(0.0, device=device, requires_grad=True)
            giou_loss = torch.tensor(0.0, device=device, requires_grad=True)
            l1_loss = torch.tensor(0.0, device=device, requires_grad=True)
        bg_bce = masked_background_bce_loss(pred_logits, indices)
        return {
            'loss': self.clf_coeff * (gce_loss + bg_bce) + self.l1_coeff * l1_loss + self.giou_coeff * giou_loss,
            'GCE': gce_loss,
            'L1': l1_loss,
            'GIoU': giou_loss,
            'BG BCE': bg_bce,
        }
    
    def training_step(self, batch, batch_idx):
        images, target_boxes, target_labels = batch["image"], batch["boxes"], batch["labels"]
        pred_boxes, pred_logits = self.model(images)
        loss = self.compute_loss(pred_boxes, pred_logits, target_boxes, target_labels)
        self.log_dict({f"loss/{loss_name}": loss[loss_name] for loss_name in loss})
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6)
        return [optimizer], [scheduler]
    
    def validation_step(self, batch, batch_idx):
        images, target_boxes, target_labels = batch["image"], batch["boxes"], batch["labels"]
        pred_boxes, pred_logits = self.model(images)
        pred_scores = torch.softmax(pred_logits, dim=-1)
        pred_labels = torch.argmax(pred_scores, dim=-1)
        
        preds = [
            {"boxes": b, "scores": s.max(dim=-1).values, "labels": l}
            for b, s, l in zip(pred_boxes, pred_scores, pred_labels)
        ]
        targets = [
            {"boxes": b, "labels": l}
            for b, l in zip(target_boxes, target_labels)
        ]
        
        self.map_metric.update(preds, targets)
    
    def on_validation_epoch_end(self):
        mAP = self.map_metric.compute()
        self.log_dict({f"val_mAP/{k}": v for k, v in mAP.items()}, sync_dist=True)
        self.map_metric.reset()


def select_and_pad(pred, target, indices_list, pad_value=-1):
    """
    Selects values from pred_boxes using indices_list and pads the result to match target_boxes shape,
    preserving gradients.

    Args:
        pred: (B, n_boxes, n) - Tensor with predicted boxes or logits (with gradients)
        target: (B, n_queries, n) - Tensor with target boxes or classes
        indices_list: List of np.arrays of varying lengths (indices per batch element)
        pad_value: Value to use for padding (default -1)

    Returns:
        selected_pred_boxes: (B, n_queries, n) - Selected and padded predicted boxes (gradients preserved)
    """
    batch_size = target.shape[0]
    device = pred.device
    dtype = pred.dtype
    selected_pred_boxes = torch.full_like(target, pad_value, device=device, dtype=dtype)  

    for b_idx in range(batch_size):
        indices = torch.tensor(indices_list[b_idx], device=device, dtype=torch.long)

        if len(indices) > 0:
            selected = pred[b_idx, indices]  # Selected boxes keep gradients
            num_selected = selected.shape[0]
            selected_pred_boxes[b_idx, :num_selected] = selected
    return selected_pred_boxes
