"""
training cifar in the contrastive learning context
"""

# load packages
import os
import argparse
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn

# load file
from utils import (
    effective_expansion,
    load_config,
    fileid,
    batch_jacobian,
)
from model import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, SimCLR
from data import random_samples_by_targets, cifar10_contrastive

# arguments
parser = argparse.ArgumentParser()

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
    "--tempreture", default=0.07, type=float, help="the tempreture in softmax"
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


def main():
    # setup model paths and result paths
    args.w = args.model
    load_model_id = fileid("simclr", args)
    save_model_id = (
        fileid("simclr_cifar10_plane", args)
        + f"_{args.target_digits[0]}_{args.target_digits[1]}_{args.target_digits[2]}"
    )
    load_model_dir = os.path.join(paths["model_dir"], load_model_id)
    save_model_dir = os.path.join(paths["model_dir"], save_model_id)
    result_dir = os.path.join(paths["result_dir"], save_model_id)
    if not os.path.exists(load_model_dir):
        raise Exception("Please run run_clr.py first")
    if not os.path.exists(save_model_dir):
        os.mkdir(save_model_dir)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # init writer
    frames = len(args.save_epochs)

    # load mnist
    if args.data == "cifar10":
        _, unaugmented_train_dataset, test_dataset = cifar10_contrastive(paths["data_dir"])
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batchsize,
            shuffle=False,  # no shuffling
            drop_last=False,  # get all
        )
        unaugmented_train_loader = torch.utils.data.DataLoader(
            unaugmented_train_dataset,
            batch_size=args.batchsize,
            shuffle=False,  # no shuffling
            drop_last=False,  # get all
        )
        # get labels
        feature_labels = []
        for _, labels in unaugmented_train_loader:
            feature_labels.append(labels)
        train_labels = torch.cat(feature_labels, dim=0)  # store at cpu
    else:
        raise NotImplementedError(f"dataset {args.data} not available")

    # get scan range
    # randomly sample target digit
    assert (
        len(args.target_digits) == 3
    ), "target digits does not have exactly two numbers"

    # sample from training set
    first_num_sample, second_num_sample, third_num_sample = random_samples_by_targets(
        test_dataset, targets=args.target_digits, seed=args.seed
    )

    # setup scan
    origin_clean = (first_num_sample + second_num_sample + third_num_sample) / 3
    origin = (first_num_sample + second_num_sample + third_num_sample) / 3
    right_vec = second_num_sample - first_num_sample
    up_vec = ((first_num_sample + second_num_sample) / 2 - origin) / (
        1 / 2 / 3 ** (1 / 2)
    )

    l = torch.linspace(args.lower, args.upper, steps=args.steps)
    y = torch.linspace(args.lower, args.upper, steps=args.steps)

    raw = torch.cartesian_prod(l, y)
    scan = origin + (
        raw[:, 0].reshape(-1, 1, 1, 1) * right_vec
        + raw[:, 1].reshape(-1, 1, 1, 1) * up_vec
    )  # orthogonal decomposition (stored at cpu first)

    # save clean mid and endpoints
    torch.save(first_num_sample, os.path.join(save_model_dir, "point_one.pt"))
    torch.save(second_num_sample, os.path.join(save_model_dir, "point_two.pt"))
    torch.save(third_num_sample, os.path.join(save_model_dir, "point_three.pt"))
    torch.save(origin_clean, os.path.join(save_model_dir, "origin.pt"))

    # store metrics
    num_samples = args.steps**2  # linearly interpolate with 64 bins
    effective_volume = torch.zeros(frames + 1, num_samples, device='cpu')
    entropy = torch.zeros(frames + 1, num_samples, device='cpu')
    predictions = torch.zeros(frames + 1, num_samples, device='cpu')

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
        backbone = ResNet18(nl=nl)
    elif args.model == "34":
        backbone = ResNet34(nl=nl)
    elif args.model == "50":
        backbone = ResNet50(nl=nl)
    elif args.model == "101":
        backbone = ResNet101(nl=nl)
    elif args.model == "152":
        backbone = ResNet152(nl=nl)
    else:
        raise NotImplementedError(f"ResNet{args.model} not supported")

    # get loss

    # send to parallel
    model = SimCLR(backbone, args.batchsize, nl=nl)
    feature_map = model.feature_map  # extract feature map
    model = model.to(device)

    # set up training
    cnt = 0

    # ===============================================
    # ----------- step 2: Volume elem ----------------
    # ===============================================
    print("===== Stage 2: Geometric Quantity ======")
    for i in args.save_epochs:
        print(f'--- epoch {i} ---')
        model.load_state_dict(torch.load(os.path.join(load_model_dir, f'simclr_model_state_dict_e{i}.pt'), map_location='cpu'))
        model.eval()
        feature_map = model.feature_map.to(device)
        with torch.no_grad():
            # get geometric quantities
            detarr = torch.zeros(num_samples).to(device)
            interpolated_features_list = []

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

                # interpolated features 
                interpolated_features_list.append(feature_map(cur_scan).cpu())  # save results on cpu

            # put back
            effective_volume[cnt] = detarr.detach().cpu()

            # get interpolated features
            interpolated_features = torch.cat(interpolated_features_list, dim=0)
        
        # get train features
        feature_img = []
        feature_map = feature_map.to(device)  # send feature map to device
        with torch.no_grad():
            for images, _ in unaugmented_train_loader:
                images = images.to(device)
                cur_feature_images = feature_map(images)
                feature_img.append(cur_feature_images)
        train_features = torch.cat(feature_img, dim=0)

        # get test features
        test_feature_list, test_label_list = [], []
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(device)
                test_feature_list.append(feature_map(imgs).cpu()) # store on cpu 
                test_label_list.append(labels)
        test_features = torch.cat(test_feature_list, dim=0).detach().cpu().numpy()
        test_labels = torch.cat(test_label_list, dim=0).detach().cpu().numpy()

        # get predictions
        lr_train_data = train_features.detach().cpu().numpy()
        lr = LogisticRegression().fit(lr_train_data, train_labels)
        downstream_train_acc = lr.score(lr_train_data, train_labels)
        downstream_test_acc = lr.score(test_features, test_labels)
        print(f"downstream logistic regression train accuracy: {downstream_train_acc}; test accuracy: {downstream_test_acc}")

        # get interpolation predictions
        interpolated_features_np = interpolated_features.detach().cpu().numpy()
        cur_predictions = lr.predict(interpolated_features_np)
        
        # put back
        predictions[cnt] = torch.tensor(cur_predictions, device='cpu')

        # get entropy 
        probabilities = lr.predict_proba(interpolated_features_np)
        cur_entropy = -(
            probabilities * np.log10(probabilities)
        ).sum(axis=1)

        # put back 
        entropy[cnt] = torch.tensor(cur_entropy, device='cpu')

        cnt += 1

        # eigenvalues
        
        # get metrics at anchor points
        with torch.no_grad():
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
                os.path.join(save_model_dir, f"first_eigvals_e{i}.pt"),
            )
            torch.save(
                second_eigvals.to("cpu"),
                os.path.join(save_model_dir, f"second_eigvals_e{i}.pt"),
            )
            torch.save(
                third_eigvals.to("cpu"),
                os.path.join(save_model_dir, f"third_eigvals_e{i}.pt"),
            )

    # save geometric evaluations
    torch.save(effective_volume.to("cpu"), os.path.join(save_model_dir, "eff_vol.pt"))
    torch.save(entropy.to("cpu"), os.path.join(save_model_dir, "entropy.pt"))
    torch.save(predictions.to("cpu"), os.path.join(save_model_dir, "predictions.pt"))


if __name__ == "__main__":
    main()
