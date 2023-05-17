"""
training cifar10
"""

# load packages
import os
import argparse
import warnings
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn

# load file
from data import cifar10, cifar10_clean, random_samples_by_targets
from model import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from utils import effective_expansion, load_config, fileid

parser = argparse.ArgumentParser()

# arguments
parser.add_argument("--data", default="cifar10", type=str, help="data tested")

# model params
parser.add_argument(
    "--model",
    default="34",
    type=str,
    help="resnet model number",
    choices=["18", "34", "50", "101", "152"],
)
parser.add_argument("--epochs", default=200, type=int, help="the number of epochs")
parser.add_argument(
    "--batchsize",
    default=12,
    type=int,
    help="size of batch; if exceeding datasize, fullbatch is used",
)
parser.add_argument("--nl", default="GELU", type=str, help="the type of nonlinearity")
parser.add_argument(
    "--print-freq", default=5, type=int, help="the freqeuncy to print loss"
)
parser.add_argument(
    "--save-epochs",
    default=[0, 50, 200],
    type=int,
    nargs="+",
    help="specific epochs to compute and save geometric quantities",
)
parser.add_argument(
    "--k", default=10, type=int, help="the number of eigenvalues to keep"
)
parser.add_argument(
    "--thr",
    default=-float("inf"),
    type=float,
    help="the threshold below which singular values are dropped in effective volume element computations; default no thresholding",
)

# opt
parser.add_argument(
    "--opt",
    default="sgd",
    type=str,
    help="the type of optimizer",
    choices=["sgd", "adam"],
)
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
parser.add_argument(
    "--scanbatchsize",
    default=40,
    type=int,
    help="the number of scan points for geometric quanities computations",
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

# device
device = torch.device(
    "cuda" if torch.cuda.is_available() and (not args.no_gpu) else "cpu"
)

# load paths
paths = load_config(tag=args.tag)

# set seed
torch.manual_seed(args.seed)

# we cannot force 64, otherwise would not fit memory of A100 even for 1 scan point
# torch.set_default_dtype(torch.float64)


def main():
    # setup model paths and result paths
    args.w = args.model  # ducktype a parameter
    load_model_id = fileid('resnet', args)
    save_model_id = (
        fileid("cifar10", args) + f"_{args.target_digits[0]}_{args.target_digits[1]}"
    )
    load_model_dir = os.path.join(paths['model_dir'], load_model_id)
    save_model_dir = os.path.join(paths["model_dir"], save_model_id)
    result_dir = os.path.join(paths["result_dir"], save_model_id)
    if not os.path.exists(load_model_dir):
        raise Exception("Please run run_resnet.py first")
    if not os.path.exists(save_model_dir):
        os.mkdir(save_model_dir)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    frames = len(args.save_epochs)

    # load mnist
    if args.data == "cifar10":
        train_set, test_set = cifar10(paths["data_dir"])
        _, test_set_clean = cifar10_clean(paths["data_dir"])
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batchsize, shuffle=True, num_workers=2
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args.batchsize, shuffle=False, num_workers=2
        )
    else:
        raise NotImplementedError(f"dataset {args.data} not available")

    # get scan range
    # randomly sample target digit
    assert (
        len(args.target_digits) == 2
    ), "target digits does not have exactly two numbers"

    # sample from testing set
    first_num_sample, second_num_sample = random_samples_by_targets(
        test_set, targets=args.target_digits, seed=args.seed
    )
    first_num_sample_clean, second_num_sample_clean = random_samples_by_targets(
        test_set_clean, targets=args.target_digits, seed=args.seed
    )  # same seed gaurantee the same image
    mid_clean = (first_num_sample_clean + second_num_sample_clean) / 2

    # setup scan from preprocessed test set
    t = torch.arange(args.steps, device=first_num_sample.device).reshape(-1, 1, 1, 1)
    scan = (second_num_sample - first_num_sample) * t / args.steps + first_num_sample
    scan = scan.to(device)  # fit the entire scan to device (should be reasonable)

    # save clean mid and endpoints
    torch.save(first_num_sample_clean, os.path.join(save_model_dir, "point_left.pt"))
    torch.save(second_num_sample_clean, os.path.join(save_model_dir, "point_right.pt"))
    torch.save(mid_clean, os.path.join(save_model_dir, "point_mid.pt"))

    # store metrics
    num_samples = args.steps  # linearly interpolate with 64 bins
    effective_volume = torch.zeros(frames + 1, num_samples, device=device)
    predictions = torch.zeros(frames + 1, num_samples, device=device)

    # get model
    # select non linearity
    if args.nl == "Sigmoid":
        nl = nn.Sigmoid()
    elif args.nl == "Erf":

        def nl(x):
            return torch.erf(x / (2 ** (1 / 2)))

    elif args.nl == "GELU":
        nl = nn.GELU()
    elif args.nl == "ELU":
        nl = nn.ELU()
    elif args.nl == "ReLU":
        nl = nn.ReLU()
        warnings.warn("Caution: ReLU is not smooth")
    else:
        raise NotImplementedError(f"nl {args.nl} not supported")

    # init model
    if args.model == "18":
        model = ResNet18(nl=nl)
    elif args.model == "34":
        model = ResNet34(nl=nl)
    elif args.model == "50":
        model = ResNet50(nl=nl)
    elif args.model == "101":
        model = ResNet101(nl=nl)
    elif args.model == "152":
        model = ResNet152(nl=nl)
    else:
        raise NotImplementedError(f"ResNet{args.model} not supported")

    # send to parallel
    model = model.to(device)

    # set up training
    cnt = 0

    # training
    for i in args.save_epochs:
        print(f'--- epoch {i} ---')
        # load model 
        model.load_state_dict(torch.load(os.path.join(load_model_dir, f'resnet_model_state_dict_e{i}.pt'), map_location=device))
        model.eval()
        feature_map = model.feature_map.to(device)

        with torch.no_grad():
            # get geometric quantities
            detarr = torch.zeros(num_samples).to(device)

            # feed to computation by batch
            num_scan_loops = int(np.ceil(num_samples / args.scanbatchsize))
            for loop in tqdm(range(num_scan_loops)):
                start_idx = loop * args.scanbatchsize
                end_idx = (loop + 1) * args.scanbatchsize
                detarr[start_idx:end_idx] = effective_expansion(
                    scan[start_idx:end_idx],
                    feature_map,
                    k=args.k,
                    thr=args.thr,
                ).squeeze()

            # put back
            effective_volume[cnt] = detarr
            cur_prediction = torch.argmax(model(scan), dim=1)  # predictions
            predictions[cnt] = cur_prediction

            # get model params
            cnt += 1

    # save geometric evaluations
    torch.save(effective_volume.to("cpu"), os.path.join(save_model_dir, "eff_vol.pt"))
    torch.save(predictions.to("cpu"), os.path.join(save_model_dir, "predictions.pt"))

if __name__ == "__main__":
    main()
