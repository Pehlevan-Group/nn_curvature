"""
visualize deeper network 
"""
# load package
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

# load file
from utils import load_config, fileid
from vis import makevid_analytic, makevid_empirical_multi

# arguments
parser = parser = argparse.ArgumentParser()

# model params
parser.add_argument(
    "--data",
    default="XOR",
    type=str,
    help="data tested",
    choices=["XOR", "noisyXOR", "sindata", "sindata_full"],
)
parser.add_argument(
    "--ws", default=[4, 4, 4], type=int, nargs="+", help="tested widths"
)
parser.add_argument("--output-dim", default=1, type=int, help="the output dimension")
parser.add_argument(
    "--nl", default="Sigmoid", type=str, help="the type of nonlinearity"
)
parser.add_argument("--epochs", default=2000, type=int, help="the number of epochs")
parser.add_argument(
    "--print-freq", default=500, type=int, help="the freqeuncy to print loss"
)
parser.add_argument(
    "--save-freq", default=50, type=int, help="the frequency to save metrics"
)

# opt
parser.add_argument("--lr", default=0.001, type=float, help="the learning rate for SGD")
parser.add_argument("--momentum", default=0.9, type=float, help="the momentum in SGD")
parser.add_argument(
    "--weight-decay", default=0, type=float, help="the weight decay for SGD"
)

# loss
parser.add_argument(
    "--loss",
    default="MSELoss",
    type=str,
    help="the type of loss in training",
    choices=["MSELoss", "CrossEntropyLoss"],
)

# visualize params
parser.add_argument("--upper", default=1.5, type=float, help="the upper bound of vis")
parser.add_argument("--lower", default=-1.5, type=float, help="the lower bound of vis")
parser.add_argument(
    "--steps", default=40, type=int, help="the number of steps to traverse in each axis"
)
parser.add_argument(
    "--plot-mode",
    default="analytic",
    type=str,
    help="the anlaytic boundary",
    choices=["analytic", "empirical"],
)
parser.add_argument(
    "--plot-line",
    action="store_true",
    default=False,
    help="True to plot decision boundary",
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
if args.plot_mode == "analytic":
    makevid = makevid_analytic
elif args.plot_mode == "empirical":
    makevid = makevid_empirical_multi
else:
    raise NotImplementedError(f"plot mode {args.plot_mode} not supported")


def main():
    # parse plot parameters
    num_hidden_layers = len(args.ws)
    args.w = "_".join([str(x) for x in args.ws])
    model_id = fileid("deepxor", args)
    result_dir = os.path.join(paths["result_dir"], model_id)
    model_dir = os.path.join(paths["model_dir"], model_id)

    # load model parameters and evaluated geometric quanities
    dettens_list = [
        torch.load(os.path.join(model_dir, f"dettens_{m + 1}.pt"), map_location=device)
        for m in range(num_hidden_layers)
    ]
    # curvtens = torch.load(os.path.join(
    #     model_dir, "curvtens.pt"), map_location=device)
    colorchange = torch.load(
        os.path.join(model_dir, "colorchange.pt"), map_location=device
    )

    # load model parameters
    weightlist = torch.load(os.path.join(model_dir, "weightlist.pt"))
    biaslist = torch.load(os.path.join(model_dir, "biaslist.pt"))

    l = torch.linspace(args.lower, args.upper, steps=args.steps).to(device)
    y = torch.linspace(args.lower, args.upper, steps=args.steps).to(device)

    frames = args.epochs // args.save_freq

    # # plot curvature growth
    # fig, ax = plt.subplots()
    # anima = makevid(
    #     w,
    #     frames,
    #     args.epochs,
    #     y,
    #     l,
    #     curvtens,
    #     colorchange,
    #     weightlist,
    #     biaslist,
    #     fig,
    #     ax,
    #     plot_line=args.plot_line,
    # )
    # # save
    # anima.save(os.path.join(
    #     result_dir, f"curvgrowth_{model_id}.gif"), writer="pillow")
    # plt.close()

    # plot expansion
    fig, ax = plt.subplots(1, num_hidden_layers, figsize=(6 * num_hidden_layers, 6))
    anima = makevid(
        None,
        frames,
        args.epochs,
        y,
        l,
        dettens_list,
        colorchange,
        weightlist,
        biaslist,
        fig,
        ax,
        plot_line=args.plot_line,
        label="volume element",
    )
    # save
    anima.save(
        os.path.join(result_dir, f"expansion_{model_id}.gif"), writer="pillow", dpi=300
    )
    plt.close()


if __name__ == "__main__":
    main()
