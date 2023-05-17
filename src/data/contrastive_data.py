"""
process and load contrastive data (SimCLR)
adapted from https://github.com/sthalles/SimCLR
"""

# load packages
from PIL import ImageOps, ImageFilter
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from torchvision.transforms import transforms

np.random.seed(0)

# ============= SimClr ===============


class GaussianBlur:
    """blur a single image on CPU"""

    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(
            3,
            3,
            kernel_size=(kernel_size, 1),
            stride=1,
            padding=0,
            bias=False,
            groups=3,
        )
        self.blur_v = nn.Conv2d(
            3,
            3,
            kernel_size=(1, kernel_size),
            stride=1,
            padding=0,
            bias=False,
            groups=3,
        )
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(nn.ReflectionPad2d(radias), self.blur_h, self.blur_v)

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img


class ContrastiveLearningViewGenerator:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        """
        :param base_transform: the pytorch transformation pipeline
        :param n_views: number of augmentations
        """
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for _ in range(self.n_views)]


def get_simclr_pipeline_transform(size, s=1):
    """
    Return a set of data augmentation transformations as described in the SimCLR paper

    :param size: crop size
    :param s: the strength of color distortion
    :return a pytorch image transformation
    """
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size=int(0.1 * size)),
            transforms.ToTensor(),
        ]
    )
    return data_transforms


# ============ Barlow =============

BARLOW_NORMLIZATION = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )

class GaussianBlurBarlow(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() < self.p:
            sigma = torch.rand(1).item() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class Transform:
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlurBarlow(p=1.0),
                Solarization(p=0.0),
                transforms.ToTensor(),
                BARLOW_NORMLIZATION,
            ]
        )
        self.transform_prime = transforms.Compose(
            [
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlurBarlow(p=0.1),
                Solarization(p=0.2),
                transforms.ToTensor(),
                BARLOW_NORMLIZATION,
            ]
        )

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2


# ============ CIFAR 10 ===========
def cifar10_contrastive(data_path: str, transformation: str = "SimClr"):
    """
    SimCLR cifar10 constrastive learning dataset

    :param transformation: SimClr/Barlow
    """

    # fetch train transformation
    if transformation == "SimClr":
        train_transformation = ContrastiveLearningViewGenerator(
            get_simclr_pipeline_transform(32), 2
        )
        test_transformation = transforms.Compose([transforms.ToTensor()])
    elif transformation == "Barlow":
        train_transformation = Transform()
        test_transformation = transforms.Compose([
            transforms.ToTensor(),
            BARLOW_NORMLIZATION
        ])
    else:
        raise NotImplementedError(
            "transformation type unknown, accept only SimClr or Barlow"
        )

    # train
    train_dataset = datasets.CIFAR10(
        data_path,
        train=True,
        transform=train_transformation,
        download=True,
    )

    # test (no noise attack)
    unaugmented_train_dataset = datasets.CIFAR10(
        data_path,
        train=True,
        transform=test_transformation,  # no augmentation
        download=True,
    )

    test_dataset = datasets.CIFAR10(
        data_path,
        train=False,
        transform=transforms.Compose([transforms.ToTensor()]),  # no augmentation
        download=True,
    )
    return train_dataset, unaugmented_train_dataset, test_dataset


class FeatureDataset(Dataset):
    """pack tensors from the feature map into another dataset for downstream evaluations"""

    def __init__(self, X: torch.Tensor, y: torch.LongTensor) -> None:
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]
