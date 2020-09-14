import torch

import cv2

import numpy as np

import albumentations as A

from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class HorizontalFlip(object):
    def __init__(self, prob):
        self.transform = A.Compose(
            [A.HorizontalFlip(p=prob)],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'], min_visibility=0.3),
        )

    def __call__(self, image, target):
        transformed = self.transform(image=image, bboxes=target['boxes'], category_ids=[0 for _ in target['boxes']])
        target['boxes'] = transformed['bboxes']
        return transformed['image'], target


class Rotate(object):
    def __init__(self, prob):
        self.transform = A.Compose(
            [A.Rotate(p=prob)],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'], min_visibility=0.3),
        )

    def __call__(self, image, target):
        transformed = self.transform(image=image, bboxes=target['boxes'], category_ids=[0 for _ in target['boxes']])
        target['boxes'] = transformed['bboxes']
        return transformed['image'], target


class RandomScale(object):
    def __init__(self, prob):
        self.transform = A.Compose(
            [A.RandomScale(p=prob)],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'], min_visibility=0.3),
        )

    def __call__(self, image, target):
        transformed = self.transform(image=image, bboxes=target['boxes'], category_ids=[0 for _ in target['boxes']])
        target['boxes'] = transformed['bboxes']
        return transformed['image'], target


class RandomBrightnessContrast(object):
    def __init__(self, prob):
        self.transform = A.Compose(
            [A.RandomBrightnessContrast(p=prob)],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'], min_visibility=0.3),
        )

    def __call__(self, image, target):
        transformed = self.transform(image=image, bboxes=target['boxes'], category_ids=[0 for _ in target['boxes']])
        target['boxes'] = transformed['bboxes']
        return transformed['image'], target


class RGBShift(object):
    def __init__(self, prob, r_shift_limit=30, g_shift_limit=30, b_shift_limit=30):
        self.transform = A.Compose(
            [A.RGBShift(p=prob, r_shift_limit=r_shift_limit, g_shift_limit=g_shift_limit, b_shift_limit=b_shift_limit)],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'], min_visibility=0.3),
        )

    def __call__(self, image, target):
        transformed = self.transform(image=image, bboxes=target['boxes'], category_ids=[0 for _ in target['boxes']])
        target['boxes'] = transformed['bboxes']
        return transformed['image'], target


class ResizeKeepDim(object):
    def __init__(self, weight, height):
        self.h = height
        self.w = weight

    def __call__(self, image, target):
        h, w, c = image.shape
        if h > self.h and w > self.w:
            rescale = min(self.h / h, self.w / w)
        elif h > self.h:
            rescale = self.h / h
        elif w > self.w:
            rescale = self.w / w
        else:
            rescale = 1.0
        t_image = image.copy()
        if rescale < 1.0:
            target['boxes'] = target['boxes'] * rescale
            t_image = cv2.resize(t_image, (int(w * rescale), int(h * rescale)))
        return self.add_padding_(t_image), target

    def add_padding_(self, image):
        t_image = np.zeros((self.h, self.w, image.shape[2]), dtype=np.uint8)
        t_image[:image.shape[0], :image.shape[1], :] = image
        return t_image


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        if len(target['boxes']) == 0:
            return image, target
        target['boxes'] = torch.stack([torch.tensor(_) for _ in target['boxes']]).float()
        return image, target
