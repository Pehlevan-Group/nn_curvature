"""visualize mnist plane"""
# load packages
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

# load file
from vis import makepic_digits_plane
from utils import load_config, fileid
from data import get_cifar_class_names

# arguments
parser = parser = argparse.ArgumentParser()

# model params
parser.add_argument(
    "--file", 
    default='cifar10_plane',
    type=str,
    help='the prefix file to read from'
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
    default=[7, 6, 1],
    nargs="+",
    type=int,
    help="the boundary digits to interpolate",
)
parser.add_argument("--upper", default=1.0, type=float, help="the upper bound of vis")
parser.add_argument("--lower", default=-1.0, type=float, help="the lower bound of vis")
parser.add_argument(
    "--steps", default=60, type=int, help="the steps to take in interpolation"
)

parser.add_argument(
    "--ternary",
    default=False,
    action="store_true",
    help="True to visualize only the convex hull of the anchor point, otherwise affine hull (i.e. the entire plane)",
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
makepic = makepic_digits_plane


def main():
    # parse plot parameters
    args.w = args.model
    model_id = (
        fileid(args.file, args)
        + f"_{args.target_digits[0]}_{args.target_digits[1]}_{args.target_digits[2]}"
    )
    result_dir = os.path.join(paths["result_dir"], model_id)
    model_dir = os.path.join(paths["model_dir"], model_id)

    # load model parameters and evaluated geometric quanities
    eff_vols = torch.load(os.path.join(model_dir, "eff_vol.pt"), map_location=device)
    predictions = torch.load(
        os.path.join(model_dir, "predictions.pt"), map_location=device
    )
    entropy_all = torch.load(os.path.join(model_dir, "entropy.pt"), map_location=device)
    point_one = torch.load(os.path.join(model_dir, "point_one.pt"), map_location=device)
    point_two = torch.load(os.path.join(model_dir, "point_two.pt"), map_location=device)
    point_three = torch.load(
        os.path.join(model_dir, "point_three.pt"), map_location=device
    )
    origin = torch.load(os.path.join(model_dir, "origin.pt"), map_location=device)

    l = torch.linspace(args.lower, args.upper, steps=args.steps).to(device)
    y = torch.linspace(args.lower, args.upper, steps=args.steps).to(device)

    # get cifar labels
    labels = get_cifar_class_names()

    # plot curvature growth
    for e, eff_vol, entropy, prediction in zip(
        args.save_epochs, eff_vols, entropy_all, predictions
    ):
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        makepic(
            labels,
            e,
            l,
            y,
            eff_vol,
            entropy,
            prediction,
            point_one,
            point_two,
            point_three,
            origin,
            fig,
            ax,
            ternary=args.ternary,
        )
        # save
        save_name = (
            f"effvols_{model_id}" + f"_ternary" * args.ternary + f"_e{e}" + ".pdf"
        )
        plt.savefig(
            os.path.join(result_dir, save_name),
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close()

    # plot eigen spectrum shifts
    for anchor_num, anchor_str in zip(args.target_digits, ["first", "second", "third"]):
        eigvals_list = [
            torch.load(
                os.path.join(model_dir, f"{anchor_str}_eigvals_e{e}.pt"),
                map_location=device,
            )
            .detach()
            .cpu()
            .numpy()
            for e in args.save_epochs
        ]
        fig, ax = plt.subplots()
        fig.set_tight_layout(True)
        for eigvals in eigvals_list:
            ax.plot(eigvals)
        plt.legend(args.save_epochs, title="epoch")
        plt.title(r"$\log(\sqrt{\lambda_i})$ at " + f"{labels[int(anchor_num)]}")
        plt.ylabel(r"$\log(\sqrt{\lambda_i})$")
        plt.savefig(
            os.path.join(
                result_dir, f"eigvals_by_epochs_{labels[int(anchor_num)]}.pdf"
            ),
            bbox_inches="tight",
            dpi=300,
            facecolor="white",
        )


if __name__ == "__main__":
    main()
