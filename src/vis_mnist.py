"""visualize mnist curve"""
"""
visualize xor network 
"""
# load packages
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

parser = parser = argparse.ArgumentParser()

# load file
from utils import load_config, fileid
from vis import makevid_digits

# arguments

# model params
parser.add_argument(
    "--data",
    default="mnist",
    type=str,
    help="data tested",
)
parser.add_argument("--w", default=30, type=int, help="tested widths")
parser.add_argument("--epochs", default=2000, type=int, help="the number of epochs")
parser.add_argument(
    "--nl", default="Sigmoid", type=str, help="the type of nonlinearity"
)
parser.add_argument(
    "--print-freq", default=500, type=int, help="the freqeuncy to print loss"
)
parser.add_argument(
    "--save-freq", default=5, type=int, help="the frequency to save metrics"
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
makevid = makevid_digits


def main():
    # parse plot parameters
    w = args.w
    model_id = (
        fileid("mnist", args) + f"_{args.target_digits[0]}_{args.target_digits[1]}"
    )
    result_dir = os.path.join(paths["result_dir"], model_id)
    model_dir = os.path.join(paths["model_dir"], model_id)

    # load model parameters and evaluated geometric quanities
    eff_vol = torch.load(os.path.join(model_dir, "eff_vol.pt"), map_location=device)
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

    frames = args.epochs // args.save_freq

    # plot curvature growth
    fig, ax = plt.subplots()
    anima = makevid(
        w,
        frames,
        args.epochs,
        eff_vol,
        predictions,
        img_left,
        img_right,
        img_mid,
        fig,
        ax,
    )
    # save
    anima.save(os.path.join(result_dir, f"effvols_{model_id}.gif"), writer="pillow", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
