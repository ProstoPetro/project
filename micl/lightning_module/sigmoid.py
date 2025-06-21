from functools import lru_cache, partial

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F
from torchvision.ops import (
    generalized_box_iou,
    generalized_box_iou_loss,
    sigmoid_focal_loss,
)

from .open_set import DetectionModule
from ..dataset.vindr import ABSENT_IN_TEST, LABELS, RARE_LABELS_TEST
from ..utils import (
    box_cxcywh_to_xyxy,
    get_dummy_tokens as get_tokens,
    get_positive_and_negative_indices,
    get_matched_boxes,
)



class DetectionModuleSigmoid(DetectionModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.focal_gamma is None:
            self.bce = nn.BCEWithLogitsLoss()
        else:
            self.bce = partial(
                sigmoid_focal_loss,
                gamma=self.focal_gamma,
                alpha=-1 if self.focal_alpha is None else self.focal_alpha,
                reduction='mean',
            )
        
    def compute_loss(self, pred_boxes, pred_logits, target_boxes, image_indices, label_indices):
        matched_indices = self.hungarian_matching(pred_boxes, target_boxes, pred_logits, image_indices, label_indices)
        device = pred_logits.device
        flattened_matched_indices = torch.Tensor([i for indices in matched_indices for i in indices]).to(torch.int)
        device = pred_logits.device
        positive_pred_boxes = get_matched_boxes(pred_boxes, matched_indices)
        pos_idx, neg_idx = get_positive_and_negative_indices(pred_logits.shape, image_indices, flattened_matched_indices.to(device))
        if len(matched_indices):
            pos_logits = pred_logits[(*pos_idx.T, label_indices)]
            # print('pos', pos_logits.shape, pred_logits.shape, pos_idx.shape, flush=True)
            positive_bce_loss = self.bce(pos_logits, torch.ones(pos_logits.shape).to(device))
            giou_loss = generalized_box_iou_loss(
                box_cxcywh_to_xyxy(positive_pred_boxes),
                box_cxcywh_to_xyxy(target_boxes),
                reduction='mean'
            )
            l1_loss = F.l1_loss(positive_pred_boxes, target_boxes)
        else:
            positive_bce_loss = torch.tensor(0.0, device=device, requires_grad=True)
            giou_loss = torch.tensor(0.0, device=device, requires_grad=True)
            l1_loss = torch.tensor(0.0, device=device, requires_grad=True)

        neg_mask = torch.ones(pred_logits.shape, dtype=bool)
        neg_mask[(*pos_idx.T, label_indices)] = False
        neg_logits = pred_logits[neg_mask]
        bg_bce = self.bce(neg_logits, torch.zeros_like(neg_logits))
        return {
            'loss': self.clf_coeff * (self.pos_weight * positive_bce_loss + bg_bce) + self.l1_coeff * l1_loss + self.giou_coeff * giou_loss,
            'positive BCE': positive_bce_loss,
            'L1': l1_loss,
            'GIoU': giou_loss,
            'BG BCE': bg_bce,
        }
        
    @torch.no_grad
    def hungarian_matching(self, batch_pred_boxes, batch_target_boxes, batch_pred_logits, image_indices, batch_label_indices):
        """ Performs the matching skipping padded boxes and labels
        """
        assert batch_target_boxes.shape[1] <= batch_pred_boxes.shape[1], (batch_target_boxes.shape, batch_pred_boxes.shape)
        bs = batch_pred_logits.shape[0]
        results = []
        for batch_idx in range(bs):
            mask = image_indices == batch_idx
            if not torch.any(mask):
                results.append(np.array([]))
                continue
            target_boxes = batch_target_boxes[mask]
            pred_boxes = batch_pred_boxes[batch_idx]
            label_indices = batch_label_indices[mask]
            
            clf = -torch.transpose(torch.sigmoid(batch_pred_logits[batch_idx, :, label_indices]), 0, 1)
            l1 = torch.cdist(target_boxes[None], pred_boxes[None], p=1)[0]
            giou = -generalized_box_iou(box_cxcywh_to_xyxy(target_boxes), box_cxcywh_to_xyxy(pred_boxes))
            cost = self.clf_cost * clf + self.giou_cost * giou + self.l1_cost * l1
            results.append(linear_sum_assignment(cost.cpu())[1])
        return results

    
    def training_step(self, batch, batch_idx):
        images, target_boxes, target_labels = batch["image"], batch["boxes"], batch["tokenized_labels"]
        image_indices, attention_mask, label_indices = batch["image_indices"], batch["attention_mask"], batch["label_indices"]
        if not target_labels.numel():
            return
        pred_boxes, pred_logits = self.model(images, target_labels, attention_mask)
        loss = self.compute_loss(pred_boxes, pred_logits, target_boxes, image_indices, label_indices)
        self.log_dict({f"loss/{loss_name}": loss[loss_name] for loss_name in loss}, prog_bar=True)
        for optimizer in self.optimizers():
            optimizer.zero_grad()
        self.manual_backward(loss['loss'])
        if self.clip_value is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_value)
        for optimizer in self.optimizers():
            optimizer.step()
        return loss
    
    @lru_cache(1)
    @torch.no_grad
    def get_dummy_text_embeddings(self, device):
        dummy_class_tokens, dummy_class_attention_masks = get_dummy_tokens(self.class_strings, self.tokenizer, device)
        return self.model.get_text_embeddings(dummy_class_tokens, dummy_class_attention_masks)
    
    def logits2scores(self, logits):
        return torch.sigmoid(logits)

    def get_pred_mask(self, pred_labels, pred_scores):
        return torch.max(pred_scores, dim=-1).values > 0.1

    def get_worthy_classes_mask(self, map):
        worthy_mask = torch.ones(len(self.class_strings)).to(torch.bool)
        worthy_mask[list(RARE_LABELS_TEST)] = False
        return worthy_mask

    def get_dummy_tokens(self, device):
        return get_tokens(list(self.class_strings), self.tokenizer, device)

    def get_labels_indices(self, batch_label_indices, batch_mask):
        return batch_label_indices[batch_mask]
