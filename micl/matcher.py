import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou

from .utils import box_cxcywh_to_xyxy


@torch.no_grad
def hungarian_matching(
    batch_pred_boxes, batch_target_boxes, batch_pred_logits, batch_target_labels,
    gce_coeff=2.0, giou_coeff=2.0, l1_coeff=5.0
):
    """ Performs the matching skipping padded boxes and labels
    Args:
        batch_pred_boxes: (batch_size, n_queries, 4)
        batch_target_boxes: (batch_size, n_boxes, 4)
        batch_pred_logits: (batch_size, n_queries, n_classes + 1)
        batch_target_labels: (batch_size, n_boxes, n_classes + 1)
    """
    assert batch_target_boxes.shape[1] <= batch_pred_boxes.shape[1], (batch_target_boxes.shape, batch_pred_boxes.shape)
    bs = batch_pred_logits.shape[0]
    results = []
    for batch_idx in range(bs):
        target_labels = batch_target_labels[batch_idx]
        mask = ~torch.all(target_labels == -1, dim=-1)
        if not torch.any(mask):
            results.append(np.array())
            continue
        target_labels = target_labels[mask]
        target_boxes = batch_target_boxes[batch_idx][mask]
        pred_logits = batch_pred_logits[batch_idx]
        pred_boxes = batch_pred_boxes[batch_idx]
        
        gce = gce_cost(target_labels, pred_logits)
        l1 = torch.cdist(target_boxes[None], pred_boxes[None], p=1)[0]
        giou = -generalized_box_iou(box_cxcywh_to_xyxy(target_boxes), box_cxcywh_to_xyxy(pred_boxes))
        cost = gce_coeff * gce + giou_coeff * giou + l1_coeff * l1
        results.append(linear_sum_assignment(cost)[1])
    return results


def gce_cost(target_labels, pred_logits):
    """
    Computes the generalized cross entropy between all queries and all targets.

    Args:
        pred_logits: (n_queries, n_classes) - Predicted logits for each query
        target_labels: (n_targets, n_classes) - One-hot encoded target labels

    Returns:
        gce_matrix: (n_targets, n_queries) - Pairwise generalized cross-entropy loss
    """
    n_queries, n_classes = pred_logits.shape
    n_targets = target_labels.shape[0]
    pred_logits_exp = pred_logits.unsqueeze(0).expand(n_targets, n_queries, n_classes)  # (n_targets, n_queries, n_classes)
    target_labels_exp = target_labels.unsqueeze(1).expand(n_targets, n_queries, n_classes)  # (n_targets, n_queries, n_classes)
    masked_logits = torch.where(target_labels_exp == 1, pred_logits_exp, torch.tensor(-float('inf'), device=pred_logits.device))
    logsumexp_pos = torch.logsumexp(masked_logits, dim=-1)  # (n_targets, n_queries)
    logsumexp_all = torch.logsumexp(pred_logits_exp, dim=-1)  # (n_targets, n_queries)
    gce_matrix = -(logsumexp_pos - logsumexp_all)  # (n_targets, n_queries)
    return gce_matrix
