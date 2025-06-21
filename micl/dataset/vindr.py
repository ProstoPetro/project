import json
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoImageProcessor

from ..clip import CLIPStuff

cv2.setNumThreads(0);
LABELS = (
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly", "Clavicle fracture", 
    "Consolidation", "Edema", "Emphysema", "Enlarged PA", "ILD", "Infiltration", "Lung Opacity", 
    "Lung cavity", "Lung cyst", "Mediastinal shift", "Nodule/Mass", "Pleural effusion", 
    "Pleural thickening", "Pneumothorax", "Pulmonary fibrosis", "Rib fracture", "Other lesion", 
    # "COPD", "Lung tumor", "Pneumonia", "Tuberculosis", "Other disease", # diseases, not lesions
)

ABSENT_IN_TEST = (6, )
RARE_LABELS_TEST = (4, 6, 7, 8, 13, 21) # <10 boxes and other lesion


class VinDr(Dataset):
    def __init__(self, train: bool, augmentations: bool = False):
        self.split = 'train' if train else 'test'
        self.path = Path(f'/home/jovyan/misha/prepared_data/micl/vindr/{self.split}')
        self.annotations = pd.read_csv(self.path / 'annotation.csv')
        self.annotations.set_index('image_id', inplace=True)
        self.ids = list(self.annotations.index)
        self.train = train
        image_processor = AutoImageProcessor.from_pretrained("microsoft/rad-dino-maira-2")
        if train and augmentations:
            self.transform = A.Compose(
                transforms=[
                    A.ToFloat(max_value=255.),
                    A.Rotate(limit=30, p=0.5, border_mode=cv2.BORDER_CONSTANT, fill=0.0),
                    A.RandomResizedCrop(size=(image_processor.crop_size['height'], image_processor.crop_size['width']), scale=(0.5625, 1.0)),
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.1),
                    A.InvertImg(p=0.1),
                    A.Normalize(mean=image_processor.image_mean[0], std=image_processor.image_std[0], max_pixel_value=1.0),
                    A.ToRGB(),
                    ToTensorV2()
                ],
                bbox_params=A.BboxParams(
                    format='albumentations', label_fields=['texts', 'labels'],
                    min_visibility=0.0625, clip=True),
            )
        else:
            height, width = image_processor.crop_size['height'], image_processor.crop_size['width']
            self.transform = A.Compose(
                transforms=[
                    A.ToFloat(max_value=255.),
                    A.Resize(height=height, width=width),
                    A.Normalize(mean=image_processor.image_mean[0], std=image_processor.image_std[0], max_pixel_value=1.0),
                    A.ToRGB(),
                    ToTensorV2()
                ],
                bbox_params=A.BboxParams(
                    format='albumentations', label_fields=['texts', 'labels'],
                    min_visibility=0.8, clip=True),
            )

    def image(self, index: int) -> torch.Tensor:
        id_ = self.ids[index]
        image = Image.open(self.path / f'{id_}.png')
        return image

    def boxes(self, index: int) -> np.ndarray:
        id_ = self.ids[index]
        if not self.train:
            return json.loads(self.annotations.loc[id_]['boxes'])['test_rad']
        
        return json.loads(self.annotations.loc[id_]['merged_boxes']) # a lists of boxes in xyxy format

    def labels(self, index: int) -> np.ndarray:
        id_ = self.ids[index]
        if not self.train:
            return json.loads(self.annotations.loc[id_]['labels'])['test_rad']
        
        all_labels = json.loads(self.annotations.loc[id_]['merged_labels']) # list of lists of text labels (multiple labels per box)
        output = []
        for labels in all_labels:
            output.append(random.choice(labels))
        return output

    def __getitem__(self, index: int) -> dict:
        labels_text = self.labels(index)
        labels = [LABELS.index(label) for label in labels_text]
        transformed = self.transform(
            image=np.array(self.image(index)),
            bboxes=self.boxes(index),
            texts=labels_text,
            labels=labels,
        )
        return {
            'image': transformed['image'],
            'boxes': torch.tensor(transformed['bboxes']),
            'string_labels': transformed['texts'],
            'labels': torch.tensor(transformed['labels']).to(torch.long),
        }

    def __len__(self) -> int:
        return len(self.ids)
