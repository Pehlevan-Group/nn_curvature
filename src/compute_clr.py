"""
compute geometric quantities of simclr (linear interpolation)
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
from data import cifar10_contrastive, random_samples_by_targets
from model import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, SimCLR
from utils import effective_expansion, load_config, fileid


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

# downstream
parser.add_argument(
    "--downstream-width",
    default=2000,
    type=int,
    help="the width of the linear downstream evaluation",
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
    args.w = args.model  # ducktype a parameter
    load_model_id = fileid("simclr", args)
    save_model_id = (
        fileid("simclr_cifar10", args)
        + f"_{args.target_digits[0]}_{args.target_digits[1]}"
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

    # initialize writer
    frames = len(args.save_epochs)

    # load data
    if args.data == "cifar10":
        train_dataset, unaugmented_train_dataset, test_dataset = cifar10_contrastive(paths["data_dir"])
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
        len(args.target_digits) == 2
    ), "target digits does not have exactly two numbers"

    # sample from testing set
    first_num_sample, second_num_sample = random_samples_by_targets(
        test_dataset, targets=args.target_digits, seed=args.seed
    )
    mid_clean = (first_num_sample + second_num_sample) / 2

    # setup scan from preprocessed test set
    t = torch.arange(args.steps, device=first_num_sample.device).reshape(-1, 1, 1, 1)
    scan = (second_num_sample - first_num_sample) * t / args.steps + first_num_sample
    # fit the entire scan to device (should be reasonable)
    scan = scan.to(device)

    # save clean mid and endpoints
    torch.save(first_num_sample, os.path.join(save_model_dir, "point_left.pt"))
    torch.save(second_num_sample, os.path.join(save_model_dir, "point_right.pt"))
    torch.save(mid_clean, os.path.join(save_model_dir, "point_mid.pt"))

    # store metrics
    num_samples = args.steps  # linearly interpolate with 64 bins
    predictions = torch.zeros(frames + 1, num_samples, device=device)
    effective_volume = torch.zeros(frames + 1, num_samples, device=device)

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

    # init model (backbone)
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

    model = SimCLR(backbone, args.batchsize, nl=nl)

    # set up training
    cnt = 0

    # ===============================================
    # ----------- step 2: Volume elem ---------------
    # ===============================================
    print("===== Stage 2: Geometric Quantity =====")
    # training
    for i in args.save_epochs:
        print(f'--- epoch {i} ---')
        # volume elements
        model.load_state_dict(torch.load(os.path.join(load_model_dir, f'simclr_model_state_dict_e{i}.pt'), map_location=device))
        model.eval()
        feature_map = model.feature_map.to(device)
        with torch.no_grad():
            # get geometric quantities
            detarr = torch.zeros(num_samples).to(device)

            # feed to computation by batch
            num_scan_loops = int(np.ceil(num_samples / args.scanbatchsize))
            for loop in range(num_scan_loops):
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

            # get interpolated features
            interpolated_features = feature_map(scan)
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

        # train predictor
        lr_train_data = train_features.detach().cpu().numpy()
        lr = LogisticRegression().fit(lr_train_data, train_labels)
        downstream_train_acc = lr.score(lr_train_data, train_labels)
        downstream_test_acc = lr.score(test_features, test_labels)
        print(f"downstream logistic regression train accuracy: {downstream_train_acc}; test accuracy: {downstream_test_acc}")

        # get interpolation predictions
        cur_predictions = lr.predict(interpolated_features.detach().cpu().numpy())

        # put back 
        predictions[cnt] = torch.tensor(cur_predictions, device=device)
        # get model params
        cnt += 1

        print(f"Finish computation for epoch {i}")

    # save geometric evaluations
    torch.save(effective_volume.to("cpu"), os.path.join(save_model_dir, "eff_vol.pt"))
    torch.save(predictions.to('cpu'), os.path.join(save_model_dir, "predictions.pt"))


if __name__ == "__main__":
    main()
