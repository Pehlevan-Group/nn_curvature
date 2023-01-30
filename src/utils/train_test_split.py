"""
for train test split
"""

# load packages
import torch


def train_test_split(n: int, test_size: float = 0.25, seed: int = 42):
    """
    perform train test split and return index

    :param n: the total number of instances
    :param test_size: the proportion of test size
    :param seed: the seed for torch random generator
    """
    # set seed
    torch.manual_seed(seed)

    # partition index
    num_tests = int(n * test_size)
    indicies = torch.randperm(n)

    train_index, test_index = indicies[:-num_tests], indicies[-num_tests:]
    return train_index, test_index
