"""
collection of real-world dataset
"""

# load packages
import numpy as np
from sklearn.datasets import load_digits

# import keras.datasets.mnist

import torch
from torch.utils.data import RandomSampler
import torchvision
import torchvision.transforms as transforms

# ================= MNIST ==================
def mnist_small():
    """MNIST hand-written digits, each image is of 8 by 8. Return flattened version"""
    raw = load_digits()
    X = raw["data"]
    Y = raw["target"]
    # convert to torch tensor
    X = torch.tensor(X)
    Y = torch.LongTensor(Y)
    return X, Y


# def mnist():
#     """MNIST hand-written digits, with large images. Return flattened version"""
#     # only load train
#     (X, Y), (_, _) = keras.datasets.mnist.load_data()
#     # reshape
#     X = X.reshape(len(X), -1)  # flatten each image
#     # convert to torch tensor
#     X = torch.tensor(X).to(torch.float64)
#     Y = torch.LongTensor(Y)
#     return X, Y


def mnist():
    """local load"""
    X = np.load("data/mnist/X.npy")
    Y = np.load("data/mnist/Y.npy")
    # reshape
    X = X.reshape(len(X), -1)  # flatten each image
    # convert to torch tensor
    X = torch.tensor(X).to(torch.float64)
    Y = torch.LongTensor(Y)
    return X, Y


# =================== CIFAR 10 ===================
def cifar10(data_path):
    """load (download if necessary) cifar10"""
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=data_path, train=True, download=True, transform=transform_train
    )

    testset = torchvision.datasets.CIFAR10(
        root=data_path, train=False, download=True, transform=transform_test
    )

    return trainset, testset


def cifar10_clean(data_path):
    """load (download if necessary) unpreprocessed cifar10 data"""
    img_transform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(
        root=data_path, train=True, download=True, transform=img_transform
    )

    testset = torchvision.datasets.CIFAR10(
        root=data_path, train=False, download=True, transform=img_transform
    )

    return trainset, testset


def get_cifar_class_names():
    """return the name of the cifar 10 classes"""
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
    return classes


def random_samples_by_targets(dataset, targets=[7, 6], seed=42):
    """
    randomly sample len(target) many instances with corresponding targets

    :param dataset: the PyTorch dataset
    :param target: the y value of samples
    :param seed: for reproducibility
    """
    torch.manual_seed(seed)
    total_num_samples = len(dataset)
    samples = []
    for cur_target in targets:
        target = None
        while cur_target != target:
            random_index = torch.randint(total_num_samples, (1,))
            data, target = dataset[random_index]
        samples.append(data)
    return samples
