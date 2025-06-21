from typing import Optional

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


from ..clip import CLIPStuff
from ..dataset.vindr import VinDr
from ..utils import box_xyxy_to_cxcywh, remove_duplicates


class VinDrDataModule(pl.LightningDataModule):
    def __init__(
            self,
            batch_size: int,
            augmentations: bool = False,
            num_workers: int = 0,
            prefetch_factor: int = 5,
            softmax: bool = False,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.augmentations = augmentations
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-general")
        self.softmax = softmax

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = VinDr(train=True, augmentations=self.augmentations)
        self.val_dataset = VinDr(train=False)
        self.test_dataset = VinDr(train=False)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
            drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size * 4,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor
        )

    def _collate_fn(self, batch):
        boxes = [d['boxes'] for d in batch]
        labels = [d['labels'] for d in batch]
        if not boxes:
            boxes = torch.Tensor([[]])
            labels = torch.Tensor([[]])
        else:
            boxes = box_xyxy_to_cxcywh(torch.cat(boxes))
            labels = torch.cat(labels)

        str_labels = [t for d in batch for t in d['string_labels']]
        str_labels, labels, label_indices = remove_duplicates(str_labels, labels)
        if self.softmax:
            str_labels.append('No pathologies')
            label_indices.append(len(str_labels) - 1)
        if not str_labels:
            tokenized_labels = torch.Tensor([[]])
            attention_mask = torch.Tensor([[]])
        else:
            tokenizer_outputs = self.tokenizer(
                str_labels,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            tokenized_labels = tokenizer_outputs['input_ids']
            attention_mask = tokenizer_outputs['attention_mask']
    
        image_indices = [i for i, d in enumerate(batch) for _ in range(len(d['boxes']))]
        image_indices = torch.tensor(image_indices, dtype=torch.int16)
        return {
            'image': torch.stack([d['image'] for d in batch]),
            'boxes': boxes,
            'tokenized_labels': tokenized_labels,
            'attention_mask': attention_mask,
            'labels': torch.tensor(labels).long(),
            'image_indices': image_indices,
            'label_indices': torch.tensor(label_indices).long(), # index of corresponding label for each box (and additional No pathologies index if softmax)
        }
