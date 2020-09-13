import os
from pathlib import Path
import cv2

import numpy as np

import torch
from torch.utils.data import Dataset


def collate(batch):
    return [_[0] for _ in batch], [_[1] for _ in batch]


class FishDataset(Dataset):
    def __init__(self, dataset_path, labels_path, transform):
        self.dataset_path = dataset_path
        self.labels_path = labels_path
        self.transform = transform
        self.init__()

    def init__(self):
        self.images_paths = Path(self.dataset_path).glob('**/*.jpg')
        self.images_paths = [str(x) for x in self.images_paths if x.is_file()]
        self.images_names = [os.path.basename(x) for x in self.images_paths]
        self.images_map = {self.images_names[i]: self.images_paths[i] for i in range(len(self.images_names))}
        with open(self.labels_path, 'r') as f:
            lines = f.readlines()
            lines = [_.strip().split(' ') for _ in lines]
        self.annotations = {_[0]: np.stack([list(map(int, bbox.split(','))) for bbox in _[1:]]) for _ in lines if
                            _[0] in self.images_map}

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.images_paths[idx])
        h, w, c = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image
        image_name = self.images_names[idx]
        boxes = self.annotations[image_name].copy()
        np.clip(boxes[:, :2], 1, None, boxes[:, :2])
        np.clip(boxes[:, 2], None, w - 1, boxes[:, 2])
        np.clip(boxes[:, 3], None, h - 1, boxes[:, 3])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.from_numpy(area)
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        image_id = torch.tensor([idx])
        anno = {'name': image_name, 'boxes': boxes, 'iscrowd': iscrowd, 'image_id': image_id, 'area': area,
                'labels': torch.ones(boxes.shape[0], dtype=torch.long)}
        return self.transform(image, anno)
