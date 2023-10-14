"""
visualize cifar linear boundary
"""

# load packages
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

# load file
from utils import load_config, fileid
from vis import makepic_digits
from data import get_cifar_class_names

# arguments
parser = parser = argparse.ArgumentParser()

# model params
parser.add_argument(
    "--file", default="cifar10_deep", type=str, help="prefix to read folder from"
)
parser.add_argument(
    "--data",
    default="cifar10",
    type=str,
    help="data tested",
)
parser.add_argument(
    "--model",
    default="34",
    type=str,
    help="resnet model number",
    choices=["18", "34", "50", "101", "152"],
)
parser.add_argument("--epochs", default=200, type=int, help="the number of epochs")
parser.add_argument("--nl", default="GELU", type=str, help="the type of nonlinearity")
parser.add_argument(
    "--print-freq", default=500, type=int, help="the freqeuncy to print loss"
)
parser.add_argument(
    "--save-epochs",
    default=[0, 50, 200],
    type=int,
    nargs="+",
    help="the frequency to save metrics",
)

# opt
parser.add_argument("--lr", default=0.001, type=float, help="the learning rate for SGD")
parser.add_argument("--momentum", default=0.9, type=float, help="the momentum in SGD")
parser.add_argument(
    "--weight-decay", default=0, type=float, help="the weight decay for SGD"
)

# digit boundary
parser.add_argument(
    "--target-digits",
    default=[7, 6],
    nargs="+",
    type=int,
    help="the boundary digits to interpolate",
)
parser.add_argument(
    "--steps", default=64, type=int, help="the steps to take in interpolation"
)

# technical
parser.add_argument(
    "--no-gpu", default=False, action="store_true", help="turn on to disable gpu usage"
)
parser.add_argument(
    "--seed", default=400, type=int, help="the random seed to run init and SGD"
)

# IO
parser.add_argument("--tag", default="exp", type=str, help="the tag for path configs")

args = parser.parse_args()

# load paths
paths = load_config(tag=args.tag)

# specify device
device = "cpu"

# ======== plotting ========

# select plotter
makepic = makepic_digits


def main():
    # parse plot parameters
    args.w = args.model  # ducktype a parameter
    model_id = (
        fileid(args.file, args) + f"_{args.target_digits[0]}_{args.target_digits[1]}"
    )
    result_dir = os.path.join(paths["result_dir"], model_id)
    model_dir = os.path.join(paths["model_dir"], model_id)

    # load model parameters and evaluated geometric quanities
    predictions = torch.load(
        os.path.join(model_dir, "predictions.pt"), map_location=device
    )
    img_left = torch.load(os.path.join(model_dir, "point_left.pt"), map_location=device)
    img_right = torch.load(
        os.path.join(model_dir, "point_right.pt"), map_location=device
    )
    img_mid = torch.load(os.path.join(model_dir, "point_mid.pt"), map_location=device)

    # load model parameters
    # weightlist = torch.load(os.path.join(model_dir, "weightlist.pt"))
    # biaslist = torch.load(os.path.join(model_dir, "biaslist.pt"))

    # get cifar10 label names
    labels = get_cifar_class_names()

    # plot curvature growth
    for i, (e, prediction) in enumerate(zip(args.save_epochs, predictions)):
        fig, ax = plt.subplots(1, 4, figsize=(24, 6))

        # read individual vol elements
        ylim_low, ylim_high = float('inf'), -float('inf')
        for layer_idx in range(4): 
            eff_vol = torch.load(os.path.join(model_dir, f'eff_vol_layer{layer_idx + 1}.pt'), map_location=device)[i]
            makepic(labels, e, eff_vol, prediction, img_left, img_right, img_mid, fig, ax[layer_idx])

            ax[layer_idx].set_title(f"Layer {layer_idx + 1}", fontsize=8)

            # adjust y lim
            cur_ylim_low, cur_ylim_high = ax[layer_idx].get_ylim()
            ylim_low = min(ylim_low, cur_ylim_low)
            ylim_high = max(ylim_high, cur_ylim_high)
        
        # readjust y lim
        for layer_idx in range(4):
            ax[layer_idx].set_ylim([ylim_low, ylim_high])

        # label epoch
        plt.suptitle(f'Epoch {e}')
        plt.savefig(
            os.path.join(result_dir, f"eff_vol_e{e}.pdf"),
            dpi=300,
            facecolor="white",
            bbox_inches="tight",
        )

if __name__ == "__main__":
    main()
