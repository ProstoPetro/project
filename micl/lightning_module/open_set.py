from functools import lru_cache, partial

import numpy as np
import torch
from lightning import LightningModule
from scipy.optimize import linear_sum_assignment
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ChainedScheduler, CosineAnnealingLR, LinearLR
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import (
    generalized_box_iou,
    generalized_box_iou_loss,
)

from ..dataset.vindr import ABSENT_IN_TEST, LABELS, RARE_LABELS_TEST
from ..losses import FocalLoss
from ..metrics import BoxRecall, GroundingMetrics
from ..utils import (
    box_cxcywh_to_xyxy,
    custom_dist_sync_fn,
    get_dummy_tokens as get_tokens,
    get_matched_boxes,
    get_positive_and_negative_indices,
)


class DetectionModule(LightningModule):
    def __init__(
        self,
        model,
        n_epochs, # for scheduler
        warmup_steps=None,
        class_strings=None,
        clf_coeff=1.0,
        giou_coeff=2.0,
        l1_coeff=5.0,
        clf_cost=2.0,
        giou_cost=2.0,
        l1_cost=5.0,
        lr=None,
        tokenizer=None,
        clip_value=None,
        focal_alpha=None,
        focal_gamma=None,
        pos_weight=1.0
    ):
        super().__init__()
        self.model = model
        self.clf_coeff = clf_coeff
        self.giou_coeff = giou_coeff
        self.l1_coeff = l1_coeff
        self.clf_cost = clf_cost
        self.giou_cost = giou_cost
        self.l1_cost = l1_cost
        if lr is None:
            lr = {'vision': 1e-5, 'text': 1e-6, 'heads': 1e-4,}
        self.lr = lr
        if warmup_steps is None:
            warmup_steps = {'vision': 20, 'text': 20, 'heads': 20,}
        self.warmup_steps = warmup_steps
        self.map_metrics = {
            'map50': MeanAveragePrecision(
                box_format='cxcywh',
                class_metrics=True,
                iou_thresholds=[0.5],
                dist_sync_fn=custom_dist_sync_fn,
            ),
            'map25': MeanAveragePrecision(
                box_format='cxcywh',
                class_metrics=True,
                iou_thresholds=[0.25],
                dist_sync_fn=custom_dist_sync_fn,
            ),
        }
        self.grounding_metrics = GroundingMetrics(thresholds=[0.25, 0.5])
        self.recall_metrics = {
            'box_recall25': BoxRecall(threshold=0.25),
            'box_recall50': BoxRecall(threshold=0.5),
        }
        if focal_gamma is None:
            self.ce = nn.CrossEntropyLoss()
        else:
            self.ce = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.tokenizer = tokenizer
        if class_strings is None:
            class_strings = LABELS
        self.class_strings = class_strings
        self.n_epochs = n_epochs
        self.warmup_steps = warmup_steps
        self.clip_value = clip_value
        self.automatic_optimization = False
        self.bg_token, self.bg_attention_mask = get_tokens(['No pathologies'], tokenizer)
        self.pos_weight = pos_weight

    def compute_loss(self, pred_boxes, pred_logits, target_boxes, image_indices, label_indices):
        matched_indices = self.hungarian_matching(pred_boxes, target_boxes, pred_logits, image_indices, label_indices)
        flattened_matched_indices = torch.Tensor([i for indices in matched_indices for i in indices]).to(torch.int)
        device = pred_logits.device
        positive_pred_boxes = get_matched_boxes(pred_boxes, matched_indices)
        pos_idx, neg_idx = get_positive_and_negative_indices(pred_logits.shape, image_indices, flattened_matched_indices.to(device))

        if len(matched_indices):
            positive_bce_loss = self.ce(
                pred_logits[tuple(pos_idx.T)],
                label_indices[:-1]
            )
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
        # print('\n\n\n', pred_logits.shape, len(image_indices), '\n\n\n', flush=True)
        bg_bce = self.ce(
            pred_logits[tuple(neg_idx.T)],
            torch.full((neg_idx.shape[0], ), label_indices[-1], dtype=torch.long).to(device)
        )
        return {
            'loss': self.clf_coeff * (self.pos_weight * positive_bce_loss + bg_bce) + self.l1_coeff * l1_loss + self.giou_coeff * giou_loss,
            'Positive BCE': positive_bce_loss,
            'L1': l1_loss,
            'GIoU': giou_loss,
            'BG BCE': bg_bce,
        }
        
    @torch.no_grad
    def hungarian_matching(self, batch_pred_boxes, batch_target_boxes, batch_pred_logits, image_indices, batch_label_indices):
        """ Performs the matching skipping padded boxes and labels"""
        assert batch_target_boxes.shape[1] <= batch_pred_boxes.shape[1], (batch_target_boxes.shape, batch_pred_boxes.shape)
        batch_pred_probs = torch.softmax(batch_pred_logits, dim=-1)[..., :-1]
        # batch_pred_probs = torch.sigmoid(batch_pred_logits)[..., :-1]

        bs = batch_pred_logits.shape[0]
        results = []
        for batch_idx in range(bs):
            mask = image_indices == batch_idx
            if not torch.any(mask):
                results.append(np.array([]))
                continue
            
            label_indices = batch_label_indices[:-1][mask]
            target_boxes = batch_target_boxes[mask]
            pred_boxes = batch_pred_boxes[batch_idx]

            clf = -torch.transpose(batch_pred_probs[batch_idx, :, label_indices], 0, 1)
            l1 = torch.cdist(target_boxes[None], pred_boxes[None], p=1)[0]
            giou = -generalized_box_iou(box_cxcywh_to_xyxy(target_boxes), box_cxcywh_to_xyxy(pred_boxes))
            cost = self.clf_cost * clf + self.giou_cost * giou + self.l1_cost * l1

            results.append(linear_sum_assignment(cost.cpu())[1])
        # return results, clf, l1, giou, cost
        return results

    def training_step(self, batch, batch_idx):
        images, target_boxes, target_labels = batch["image"], batch["boxes"], batch["tokenized_labels"]
        image_indices, attention_mask, label_indices = batch["image_indices"], batch["attention_mask"], batch["label_indices"]
        if target_boxes.numel() == 0:
            loss = {'loss': torch.tensor(0.0, device=self.device, requires_grad=True)}
        else:
            pred_boxes, pred_logits = self.model(images, target_labels, attention_mask)
            loss = self.compute_loss(pred_boxes, pred_logits, target_boxes, image_indices, label_indices)
        self.log_dict({f"loss/{loss_name}": loss[loss_name] for loss_name in loss}, prog_bar=True, rank_zero_only=True)
        for optimizer in self.optimizers():
            optimizer.zero_grad()
        self.manual_backward(loss['loss'])
        if self.clip_value is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_value)
        for optimizer in self.optimizers():
            optimizer.step()
        return loss

    def on_train_epoch_end(self):
        for scheduler in self.lr_schedulers():
            scheduler.step()
    
    def configure_optimizers(self):
        heads_params = list(self.model.clf_head.parameters()) + list(self.model.box_head.parameters())
        optimizer1 = optim.AdamW(self.model.vision_model.parameters(), lr=self.lr['vision'], weight_decay=1e-4)
        optimizer2 = optim.AdamW(self.model.text_model.parameters(), lr=self.lr['text'], weight_decay=1e-4)
        optimizer3 = optim.AdamW(heads_params, lr=self.lr['heads'], weight_decay=1e-4)
        scheduler1 = ChainedScheduler([
            LinearLR(optimizer1, start_factor=1 / (self.warmup_steps['vision'] + 1), end_factor=1.0, total_iters=self.warmup_steps['vision']),
            CosineAnnealingLR(optimizer1, T_max=self.n_epochs, eta_min=self.lr['vision'] * 0.2),
        ], optimizer1)
        scheduler2 = ChainedScheduler([
            LinearLR(optimizer2, start_factor=1 / (self.warmup_steps['text'] + 1), end_factor=1.0, total_iters=self.warmup_steps['text']),
            CosineAnnealingLR(optimizer2, T_max=self.n_epochs, eta_min=self.lr['text'] * 0.2),
        ], optimizer2)
        scheduler3 = ChainedScheduler([
            LinearLR(optimizer3, start_factor=1 / (self.warmup_steps['heads'] + 1), end_factor=1.0, total_iters=self.warmup_steps['heads']),
            CosineAnnealingLR(optimizer3, T_max=self.n_epochs, eta_min=self.lr['heads'] * 0.2),
        ], optimizer3)
        return [optimizer1, optimizer2, optimizer3], [scheduler1, scheduler2, scheduler3]
        # return optim.AdamW(self.model.parameters(), lr=self.lr['vision'], weight_decay=1e-4)
    
    @lru_cache(1)
    @torch.no_grad
    def get_dummy_text_embeddings(self, device):
        dummy_class_tokens, dummy_class_attention_masks = self.get_dummy_tokens(device)
        return self.model.get_text_embeddings(dummy_class_tokens, dummy_class_attention_masks)

    def logits2scores(self, logits):
        return torch.softmax(logits, dim=-1)

    def get_labels_indices(self, batch_label_indices, batch_mask):
        return batch_label_indices[:-1][batch_mask]

    def get_pred_mask(self, pred_labels, pred_scores):
        return pred_labels != pred_scores.shape[-1] - 1

    def get_dummy_tokens(self, device):
        return get_tokens(
            list(self.class_strings) + ['No pathologies'],
            self.tokenizer, device
        )
    def validation_step_vindr(self, batch):
        images, target_boxes, target_tokens = batch["image"], batch["boxes"], batch["tokenized_labels"]
        batch_size = images.shape[0]
        device = images.device
        image_indices, attention_mask, target_labels = batch["image_indices"], batch["attention_mask"], batch['labels']
        batch_label_indices = batch["label_indices"]
        preds, targets = [], []
        dummy_class_tokens, dummy_class_attention_masks = self.get_dummy_tokens(device)
        batch_pred_boxes, dummy_classes_logits = self.model(images, dummy_class_tokens, dummy_class_attention_masks)

        if target_boxes.numel() == 0:
            val_loss = {'loss': torch.tensor(0.0, device=self.device, requires_grad=False)}
        else:
            val_loss = self.compute_loss(batch_pred_boxes, dummy_classes_logits, target_boxes, image_indices, batch_label_indices)
        self.log_dict({f"val_loss_vindr/{loss_name}": val_loss[loss_name] for loss_name in val_loss}, prog_bar=True, rank_zero_only=True)

        for batch_idx in range(batch_size):
            classes_logits = dummy_classes_logits[batch_idx]
            pred_boxes = batch_pred_boxes[batch_idx]
            batch_mask = image_indices == batch_idx
            label_indices = self.get_labels_indices(batch_label_indices, batch_mask)
            labels = target_labels[label_indices]

            pred_scores = self.logits2scores(classes_logits)
            pred_labels = torch.argmax(pred_scores, dim=-1)
            pred_mask = self.get_pred_mask(pred_labels, pred_scores)

            preds.append({
                "boxes": pred_boxes[pred_mask],
                "scores": pred_scores.max(dim=-1).values[pred_mask],
                "labels": pred_labels[pred_mask]
            })
            if target_tokens.numel():
                targets.append({
                    "boxes": target_boxes[batch_mask],
                    "labels": labels
                })
            else:
                targets.append({
                    "boxes": torch.Tensor([]).to(device),
                    "labels": torch.Tensor([]).to(device),
                })
        preds.append({
            "boxes": torch.Tensor([[-1, -1, 0.5, 0.5] for _ in ABSENT_IN_TEST]).to(device),
            "scores": torch.Tensor([1 for _ in ABSENT_IN_TEST]).to(device),
            "labels": torch.Tensor(ABSENT_IN_TEST).to(device).to(torch.long)
        })
        targets.append({
            'boxes': torch.Tensor([[-1, -1, 0.5, 0.5] for _ in ABSENT_IN_TEST]).to(device),
            'labels': torch.Tensor(ABSENT_IN_TEST).to(device).to(torch.long),
        })
        for metric in self.map_metrics:
            self.map_metrics[metric].update(preds, targets)
        for metric_name in self.recall_metrics:
            self.recall_metrics[metric_name].update(
                box_cxcywh_to_xyxy(batch_pred_boxes),
                box_cxcywh_to_xyxy(target_boxes),
                image_indices,
            )

    def validation_step_mscxr(self, batch):
        images, batch_target_boxes, batch_target_tokens = batch["image"], batch["boxes"], batch["tokenized_labels"]
        batch_size = images.shape[0]

        image_indices, attention_mask = batch["image_indices"], batch["attention_mask"]
        batch_label_indices = batch["label_indices"]
        batch_pred_boxes, batch_pred_logits = self.model(images, batch_target_tokens, attention_mask)

        if batch_target_boxes.numel() == 0:
            val_loss = {'loss': torch.tensor(0.0, device=self.device, requires_grad=False)}
        else:
            val_loss = self.compute_loss(batch_pred_boxes, batch_pred_logits,batch_target_boxes, image_indices, batch_label_indices)
        self.log_dict({f"val_loss_mscxr/{loss_name}": val_loss[loss_name] for loss_name in val_loss},prog_bar=True, rank_zero_only=True)
        for batch_idx in range(batch_size):
            mask = image_indices == batch_idx
            if not torch.any(mask):
                continue
            
            label_indices = batch_label_indices[:-1][mask]
            target_boxes = batch_target_boxes[mask]
            pred_boxes = batch_pred_boxes[batch_idx]
            pred_logits = batch_pred_logits[batch_idx]
            for label in set(label_indices):
                pred_mask = pred_logits[:, label] >= pred_logits[:, -1]
                self.grounding_metrics.update(
                    box_cxcywh_to_xyxy(pred_boxes[pred_mask]),
                    box_cxcywh_to_xyxy(target_boxes[label_indices == label]),
                )

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            self.validation_step_vindr(batch)
        elif dataloader_idx == 1:
            self.validation_step_mscxr(batch)

    def get_worthy_classes_mask(self, map):
        worthy_mask = torch.ones(len(map['map_per_class'])).to(torch.bool)
        worthy_mask[list(RARE_LABELS_TEST)] = False
        if len(worthy_mask) > len(self.class_strings):
            worthy_mask[-1] = False
        return worthy_mask

    def on_validation_epoch_end(self):
        map = {metric_name: metric.compute() for metric_name, metric in self.map_metrics.items()}
        grounding = self.grounding_metrics.compute()
        if not self.trainer.sanity_checking:
            metrics_dict = {}
            for metric in self.map_metrics:
                worthy_mask = self.get_worthy_classes_mask(map[metric])
                metrics_dict.update({
                    f'val/all_classes_{metric}': map[metric]['map'].to(self.device),
                    f'val/worthy_classes_{metric}': map[metric]['map_per_class'][worthy_mask].mean().to(self.device)
                })
                # print(map[metric]['classes'], '\n\n')
                for i, label in enumerate(LABELS):
                    self.logger.experiment.add_scalar(f'detailed_metrics/{metric}/{label}', map[metric]['map_per_class'][i], self.global_step)
                # metrics_dict.update({f'val/{metric}_{label}': map[metric]['map_per_class'][i] for i, label in enumerate(LABELS)})
                self.map_metrics[metric].reset()

            for metric in self.recall_metrics:
                metrics_dict[f'val/{metric}'] = self.recall_metrics[metric].compute().to(self.device)
                self.recall_metrics[metric].reset()
            if isinstance(grounding, dict):
                metrics_dict.update({f'val/{metric}': grounding[metric] for metric in grounding})
            self.grounding_metrics.reset()
            self.log_dict(metrics_dict, sync_dist=True)
