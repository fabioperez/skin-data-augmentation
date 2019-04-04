import math
import random

from imgaug import augmenters as iaa
import imgaug as ia
import numpy as np
import PIL
import torch
from torchvision import transforms

from .tps.tps_warp import tps_warp
from .inception_crop import InceptionCrop
from .autoaugment import ImageNetPolicy


def set_seeds(worker_id):
    """
    Set random seeds. Used for setting different seeds for each
    worker created by DataLoader.
    """
    seed = torch.initial_seed() % 2**31
    ia.seed(seed + 1)
    np.random.seed(seed + 2)
    random.seed(seed + 3)


class RandomErasing:
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by
    Zhong et al.

    Modified from https://github.com/zhunzhong07/Random-Erasing

    Args:
        probability: The probability that the operation will be performed.
        sl: min erasing area
        sh: max erasing area
        r1: min aspect ratio
    '''
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(10):
            area = img.shape[0] * img.shape[1]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.shape[1] and h < img.shape[0]:
                x1 = random.randint(0, img.shape[0] - h)
                y1 = random.randint(0, img.shape[1] - w)
                if img.shape[2] == 3:
                    img[x1:x1+h, y1:y1+w, :] = np.random.rand(h, w, 3)*255.
                else:
                    img[x1:x1+h, y1:y1+w, 0] = np.random.rand(h, w, 1)*255.
                return img

        return img


class Augmentations:
    def __init__(self, **augs):
        self.mean = augs['mean']
        self.std = augs['std']
        self.size = augs['size']

        tf_list = []

        if not augs['scale']:
            augs['scale'] = 1.0

        affine = iaa.Affine(
            rotate=(-augs['rotation'], augs['rotation']),
            shear=(-augs['shear'], augs['shear']),
            scale=({'x': augs['scale'], 'y': augs['scale']}),
            mode='symmetric')

        piecewise_affine = iaa.PiecewiseAffine(
            scale=(0.0, 0.1), nb_rows=4, nb_cols=4,
            mode='symmetric')

        if augs['random_crop']:
            tf_list.append(transforms.RandomResizedCrop(
                augs['size'], scale=(0.4, 1.0)))
        else:
            tf_list.append(transforms.Resize((augs['size'], augs['size'])))
        if augs['autoaugment']:
            tf_list.append(ImageNetPolicy())
        tf_list.append(lambda x: np.array(x))
        if augs['random_erasing']:
            tf_list.append(RandomErasing(sh=0.3))
        if augs['rotation'] or augs['shear'] or augs['scale'] != 1.0:
            tf_list.append(lambda x: affine.augment_image(x))
        if augs['piecewise_affine']:
            tf_list.append(lambda x: piecewise_affine.augment_image(x))
        if augs['tps']:
            tf_list.append(lambda x: tps_warp(x, 4, 0.1))
        tf_list.append(lambda x: PIL.Image.fromarray(x))
        if augs['hflip']:
            tf_list.append(transforms.RandomHorizontalFlip())
        if augs['vflip']:
            tf_list.append(transforms.RandomVerticalFlip())
        if (augs['color_saturation'] or augs['color_contrast']
                or augs['color_brightness'] or augs['color_hue']):
            tf_list.append(transforms.ColorJitter(
                brightness=augs['color_brightness'],
                contrast=augs['color_contrast'],
                saturation=augs['color_saturation'],
                hue=augs['color_hue']))

        tf_list.append(transforms.ToTensor())
        self.tf_augment = transforms.Compose(tf_list)
        self.tf_transform = transforms.Compose([
            self.tf_augment,
            transforms.Normalize(augs['mean'], augs['std'])
        ])
        self.no_augmentation = transforms.Compose([
            transforms.Resize((augs['size'], augs['size'])),
            transforms.ToTensor(),
            transforms.Normalize(augs['mean'], augs['std'])
        ])
        self.ten_crop = self._get_crop_transform('ten')
        self.inception_crop = self._get_crop_transform('inception')

    def seed(self, seed):
        ia.seed(seed + 1 % 2**32)
        np.random.seed(seed + 1 % 2**32)
        random.seed(seed + 1 % 2**32)

    def _get_crop_transform(self, method):
        if method == 'ten':
            crop_tf = transforms.Compose([
                transforms.Resize((self.size + 32, self.size + 32)),
                transforms.TenCrop((self.size, self.size))
            ])
        if method == 'inception':
            crop_tf = InceptionCrop(
                self.size,
                resizes=tuple(range(self.size + 32, self.size + 129, 32))
            )
        after_crop = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
        return transforms.Compose([
            crop_tf,
            transforms.Lambda(
                lambda crops: torch.stack(
                    [after_crop(crop) for crop in crops]))
        ])
