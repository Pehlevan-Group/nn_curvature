"""
training cifar and store plane information
"""

# load packages
import os
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

# load file
from utils import (
    effective_expansion,
    load_config,
    fileid,
    batch_jacobian,
)
from model import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from data import cifar10, cifar10_clean, random_samples_by_targets

# arguments
parser = parser = argparse.ArgumentParser()

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
parser.add_argument(
    "--eigvals-epochs",
    default=[0, 50, 200],
    type=int,
    nargs="+",
    help="the epochs to record eigenvalues of anchors",
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

# # force 64
# torch.set_default_dtype(torch.float64)

# convert to set
save_epochs = set(args.save_epochs)
eigvals_epochs = set(args.eigvals_epochs)


def main():
    # setup model paths and result paths
    args.w = args.model
    model_id = (
        fileid("cifar10_plane", args)
        + f"_{args.target_digits[0]}_{args.target_digits[1]}_{args.target_digits[2]}"
    )
    model_dir = os.path.join(paths["model_dir"], model_id)
    result_dir = os.path.join(paths["result_dir"], model_id)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    frames = len(args.save_epochs)

    weightlist = []
    biaslist = []

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
        len(args.target_digits) == 3
    ), "target digits does not have exactly two numbers"

    # sample from training set
    first_num_sample, second_num_sample, third_num_sample = random_samples_by_targets(
        test_set, targets=args.target_digits, seed=args.seed
    )
    (
        first_num_sample_clean,
        second_num_sample_clean,
        third_num_sample_clean,
    ) = random_samples_by_targets(
        test_set_clean, targets=args.target_digits, seed=args.seed
    )
    # setup scan
    origin_clean = (
        first_num_sample_clean + second_num_sample_clean + third_num_sample_clean
    ) / 3
    origin = (first_num_sample + second_num_sample + third_num_sample) / 3
    right_vec = second_num_sample - first_num_sample
    up_vec = ((first_num_sample + second_num_sample) / 2 - origin) / (
        1 / 2 / 3 ** (1 / 2)
    )

    l = torch.linspace(args.lower, args.upper, steps=args.steps)
    y = torch.linspace(args.lower, args.upper, steps=args.steps)

    raw = torch.cartesian_prod(l, y)
    scan = (
        raw[:, 0].reshape(-1, 1, 1, 1) * right_vec
        + raw[:, 1].reshape(-1, 1, 1, 1) * up_vec
    )  # orthogonal decomposition (stored at cpu first)

    # save clean mid and endpoints
    torch.save(first_num_sample_clean, os.path.join(model_dir, "point_one.pt"))
    torch.save(second_num_sample_clean, os.path.join(model_dir, "point_two.pt"))
    torch.save(third_num_sample_clean, os.path.join(model_dir, "point_three.pt"))
    torch.save(origin_clean, os.path.join(model_dir, "origin.pt"))

    # store metrics
    num_samples = args.steps**2  # linearly interpolate with 64 bins
    effective_volume = torch.zeros(frames + 1, num_samples, device=device)
    entropy = torch.zeros(frames + 1, num_samples, device=device)
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

    # init models
    loss_func = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)

    # send to parallel
    feature_map = model.feature_map  # extract feature map
    model = model.to(device)
    if device.type == "cuda":
        model = nn.DataParallel(model)
        cudnn.benchmark = True  # reproducibility

    # select optimizer
    if args.opt == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.opt == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
            # ? amsgrad ?
        )
    else:
        raise NotImplementedError(f"opt type {args.opt} not available")

    # set up training
    epochs = args.epochs + 1
    cnt = 0
    weightlist = []
    biaslist = []

    # training
    for i in range(epochs):
        model.train()

        train_loss = 0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, targets)
            loss.backward()
            optimizer.step()

            # accuracy
            with torch.no_grad():
                train_loss += loss.item()
                predicted = outputs.argmax(dim=1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        train_acc = correct / total

        if i % args.print_freq == 0:
            model.eval()
            # get test accuracy
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = loss_func(outputs, targets)

                    test_loss += loss.item()
                    predicted = outputs.argmax(dim=1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            test_acc = correct / total

            print(
                "Epoch: {:04}, Train Loss: {:.5f}, Train Acc: {:.5f}, Test Loss: {:.5f} Test Acc: {:.5f}".format(
                    i, train_loss, train_acc, test_loss, test_acc
                )
            )

        if i in args.save_epochs:
            model.eval()
            with torch.no_grad():
                # get geometric quantities
                detarr = torch.zeros(num_samples).to(device)
                cur_entropies = torch.zeros(num_samples).to(device)
                cur_predictions = torch.zeros(num_samples).to(device)

                # feed to computation by batch
                num_scan_loops = int(np.ceil(num_samples / args.scanbatchsize))
                for loop in range(num_scan_loops):
                    # find out scan range
                    start_idx = loop * args.scanbatchsize
                    end_idx = (loop + 1) * args.scanbatchsize
                    cur_scan = scan[start_idx:end_idx].to(device)

                    # expansion
                    detarr[start_idx:end_idx] = effective_expansion(
                        cur_scan,
                        feature_map,
                        k=args.k,
                        thr=args.thr,
                    ).squeeze()

                    # prediction
                    cur_outputs = model(cur_scan)
                    cur_outputs_softmax = softmax(cur_outputs)

                    # use log10 to compute entropy so that it ranges from 0 to 1 in a 10 class classification task
                    cur_entropy = -(
                        cur_outputs_softmax * cur_outputs_softmax.log10()
                    ).sum(dim=1)

                    cur_entropies[start_idx:end_idx] = cur_entropy
                    cur_predictions[start_idx:end_idx] = torch.argmax(
                        cur_outputs, dim=1
                    )

                # put back
                effective_volume[cnt] = detarr
                predictions[cnt] = cur_predictions
                entropy[cnt] = cur_entropies

                # get model params
                weightlist.append(list(model.parameters())[0].clone())
                biaslist.append(list(model.parameters())[1].clone())
                cnt += 1

        # eigenvalues
        if i in eigvals_epochs:
            # get metrics at anchor points
            first_J = batch_jacobian(
                feature_map,
                first_num_sample.reshape(1, *first_num_sample.shape).to(device),
            )
            first_width = first_J.shape[0]
            first_J = first_J.flatten(start_dim=2).permute(1, 2, 0) / first_width ** (
                1 / 2
            )  # manual normalization
            first_eigvals = torch.linalg.svdvals(first_J).flatten().log10()

            second_J = batch_jacobian(
                feature_map,
                second_num_sample.reshape(1, *second_num_sample.shape).to(device),
            )
            second_width = second_J.shape[0]
            second_J = second_J.flatten(start_dim=2).permute(
                1, 2, 0
            ) / second_width ** (
                1 / 2
            )  # manual normalization
            second_eigvals = torch.linalg.svdvals(second_J).flatten().log10()

            third_J = batch_jacobian(
                feature_map,
                third_num_sample.reshape(1, *third_num_sample.shape).to(device),
            )
            third_width = third_J.shape[0]
            third_J = third_J.flatten(start_dim=2).permute(1, 2, 0) / third_width ** (
                1 / 2
            )  # manual normalization
            third_eigvals = torch.linalg.svdvals(third_J).flatten().log10()

            # save
            torch.save(
                first_eigvals.to("cpu"),
                os.path.join(model_dir, f"first_eigvals_e{i}.pt"),
            )
            torch.save(
                second_eigvals.to("cpu"),
                os.path.join(model_dir, f"second_eigvals_e{i}.pt"),
            )
            torch.save(
                third_eigvals.to("cpu"),
                os.path.join(model_dir, f"third_eigvals_e{i}.pt"),
            )

    # save geometric evaluations
    torch.save(model.to("cpu"), os.path.join(model_dir, "model.pt"))
    torch.save(effective_volume.to("cpu"), os.path.join(model_dir, "eff_vol.pt"))
    torch.save(entropy.to("cpu"), os.path.join(model_dir, "entropy.pt"))
    torch.save(predictions.to("cpu"), os.path.join(model_dir, "predictions.pt"))

    # save model parameters
    torch.save(weightlist, os.path.join(model_dir, "weightlist.pt"))
    torch.save(biaslist, os.path.join(model_dir, "biaslist.pt"))


if __name__ == "__main__":
    main()
