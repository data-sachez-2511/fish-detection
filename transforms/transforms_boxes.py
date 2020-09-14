import torch

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


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        if len(target['boxes']) == 0:
            return image, target
        target['boxes'] = torch.stack([torch.tensor(_) for _ in target['boxes']]).float()
        return image, target