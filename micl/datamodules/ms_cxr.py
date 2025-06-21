from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image

cv2.setNumThreads(0);
import albumentations as A
import lightning.pytorch as pl
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor

from micl.clip import CLIPStuff
from micl.image_processor import ImageProcessor

LABELS = (
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Atelectasis',
    'Edema', 'Cardiomegaly', 'Lung Opacity', 'Pleural Effusion'
)


class MSCXRDataModule(pl.LightningDataModule):
    def __init__(
            self,
            mimic_cxr_jpg_dir: str | Path,
            ms_cxr_dir: str | Path,
            batch_size: int,
            num_workers: int = 0
    ) -> None:
        super().__init__()

        self.mimic_cxr_jpg_dir = Path(mimic_cxr_jpg_dir)
        self.ms_cxr_dir = Path(ms_cxr_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.tokenizer = CLIPStuff().tokenizer

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = MSCXRDataset(self.mimic_cxr_jpg_dir, self.ms_cxr_dir, split='train')
        self.val_dataset = MSCXRDataset(self.mimic_cxr_jpg_dir, self.ms_cxr_dir, split='val')
        self.test_dataset = MSCXRDataset(self.mimic_cxr_jpg_dir, self.ms_cxr_dir, split='test')

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True
        )

    def _collate_fn(self, batch):
        str_labels = [t for d in batch for t in d['string_labels']]
        if not len(str_labels):
            return {
                'image': torch.stack([d['image'] for d in batch]),
                'boxes': torch.Tensor([]),
                'tokenized_labels': torch.Tensor([]),
                'attention_mask': torch.Tensor([]),
                'labels': torch.Tensor([]),
                'image_indices': torch.Tensor([]),
            }

        tokenizer_outputs = self.tokenizer(
            str_labels,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        image_indices = [i for i, d in enumerate(batch) for _ in range(len(d['boxes']))]
        image_indices = torch.tensor(image_indices, dtype=torch.int16)
        return {
            'image': torch.stack([d['image'] for d in batch]),
            'boxes': torch.cat([d['boxes'] for d in batch]),
            'tokenized_labels': tokenizer_outputs['input_ids'],
            'attention_mask': tokenizer_outputs['attention_mask'],
            'labels': torch.cat([d['labels'] for d in batch]),
            'image_indices': image_indices
        }


class MSCXRDataset(Dataset):
    def __init__(
            self,
            mimic_cxr_jpg_dir: Path,
            ms_cxr_dir: Path,
            split: Literal['train', 'val', 'test'],
            labels_offset: int = 0,
            multiplier: int = 1,
    ) -> None:
        super().__init__()

        self.labels_offset = labels_offset
        self.mimic_cxr_jpg_dir = Path(mimic_cxr_jpg_dir)
        self.ms_cxr_dir = Path(ms_cxr_dir)

        annotations = pd.read_csv(self.ms_cxr_dir / 'MS_CXR_Local_Alignment_v1.1.0.csv')
        annotations = annotations[annotations['split'] == split]
        self.annotations_per_dicom = sorted(annotations.groupby('dicom_id'))
        self.multiplier = multiplier
        self.split = split
        self.annotations = annotations

        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/rad-dino-maira-2")

        if split in ['train', 'val']:
            self.transform = A.Compose(
                transforms=[
                    A.ToFloat(max_value=255.),
                    A.Rotate(limit=30, p=0.5, border_mode=cv2.BORDER_CONSTANT, fill=0.0),
                    A.RandomResizedCrop(size=(self.image_processor.crop_size['height'], self.image_processor.crop_size['width']), scale=(0.5625, 1.0)),
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.1),
                    A.InvertImg(p=0.1),
                    A.Normalize(mean=self.image_processor.image_mean[0], std=self.image_processor.image_std[0], max_pixel_value=1.0),
                    A.ToRGB(),
                    ToTensorV2()
                ],
                bbox_params=A.BboxParams(format='albumentations', label_fields=['texts', 'labels'],
                                         min_visibility=0.0625, clip=True),
            )
        else:
            height, width = self.image_processor.crop_size['height'], self.image_processor.crop_size['width']
            self.transform = A.Compose(
                transforms=[
                    A.ToFloat(max_value=255.),
                    A.Resize(height=height, width=width),
                    A.Normalize(mean=self.image_processor.image_mean[0], std=self.image_processor.image_std[0], max_pixel_value=1.0),
                    A.ToRGB(),
                    ToTensorV2()
                ],
                bbox_params=A.BboxParams(format='albumentations', label_fields=['texts', 'labels'],
                                        min_visibility=0.8, clip=True),
            )

    def __len__(self) -> int:
        if self.split == 'test':
            return len(self.annotations_per_dicom)
        return len(self.annotations_per_dicom) * self.multiplier
    
    def __getitem__(self, index: int) -> dict:
        assert index < len(self.annotations_per_dicom) * self.multiplier
        index = index % len(self.annotations_per_dicom)
        _, annotations = self.annotations_per_dicom[index]

        image_path, = set(annotations['path'])
        image = Image.open(self.mimic_cxr_jpg_dir / image_path)
        boxes = []
        for _, annotation in annotations.iterrows():
            x0 = annotation['x'] / annotation['image_width']
            y0 = annotation['y'] / annotation['image_height']
            w = annotation['w'] / annotation['image_width']
            h = annotation['h'] / annotation['image_height']
            boxes.append([x0, y0, x0 + w, y0 + h])
        texts = [a['label_text'] for _, a in annotations.iterrows()]
        labels = [a['category_name'] for _, a in annotations.iterrows()]

        transformed = self.transform(image=np.array(image), bboxes=boxes, texts=texts, labels=labels)

        image = transformed['image']
        boxes = torch.tensor(transformed['bboxes'])
        texts = transformed['texts']
        labels = torch.tensor(list(map(LABELS.index, transformed['labels'])), dtype=torch.long)

        return {
            'image': image,
            'boxes': boxes, # xyxy format
            'string_labels': texts,
            'labels': torch.tensor([label + self.labels_offset for label in labels]).to(torch.long),
        }
