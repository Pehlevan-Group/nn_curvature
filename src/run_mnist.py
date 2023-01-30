"""
training MNIST 
"""

# load packages
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

parser = parser = argparse.ArgumentParser()

# load file
from data import mnist, mnist_small
from model import MLPSingle, weights_init
from utils import effective_expansion, load_config, fileid, train_test_split

# arguments
parser.add_argument("--data", default="mnist", type=str, help="data tested")

# model params
parser.add_argument("--w", default=30, type=int, help="tested widths")
parser.add_argument("--epochs", default=2000, type=int, help="the number of epochs")
parser.add_argument(
    "--batchsize",
    default=12,
    type=int,
    help="size of batch; if exceeding datasize, fullbatch is used",
)
parser.add_argument(
    "--nl", default="Sigmoid", type=str, help="the type of nonlinearity"
)
parser.add_argument(
    "--print-freq", default=5, type=int, help="the freqeuncy to print loss"
)
parser.add_argument(
    "--save-freq", default=5, type=int, help="the frequency to save metrics"
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
    default="adam",
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

# force 64
torch.set_default_dtype(torch.float64)


def main():
    # setup model paths and result paths
    model_id = (
        fileid("mnist", args) + f"_{args.target_digits[0]}_{args.target_digits[1]}"
    )
    model_dir = os.path.join(paths["model_dir"], model_id)
    result_dir = os.path.join(paths["result_dir"], model_id)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    frames = args.epochs // args.save_freq

    weightlist = []
    biaslist = []

    # load mnist
    if args.data == "mnist":
        X, Y = mnist()
    elif args.data == "mnist_small":
        X, Y = mnist_small()
    else:
        raise NotImplementedError(f"dataset {args.dataset} not available")
    X, Y = X.to(device), Y.to(device)

    # train test split
    n = X.shape[0]
    train_idx, test_idx = train_test_split(n, seed=args.seed)
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    # get scan range
    # randomly sample target digit
    assert (
        len(args.target_digits) == 2
    ), "target digits does not have exactly two numbers"
    first_num_subset = X[Y == args.target_digits[0]]
    first_num_sample = first_num_subset[[torch.randperm(first_num_subset.shape[0])[0]]]
    second_num_subset = X[Y == args.target_digits[1]]
    second_num_sample = second_num_subset[
        [torch.randperm(second_num_subset.shape[0])[0]]
    ]
    # setup scan
    t = torch.arange(args.steps, device=device).reshape(-1, 1)
    scan = (second_num_sample - first_num_sample) * t / args.steps + first_num_sample

    # save mid and endpoints
    torch.save(scan[0], os.path.join(model_dir, "point_left.pt"))
    torch.save(scan[-1], os.path.join(model_dir, "point_right.pt"))
    torch.save(scan[args.steps // 2], os.path.join(model_dir, "point_mid.pt"))

    # run for each w
    w = args.w
    # store metrics
    num_samples = args.steps  # linearly interpolate with 64 bins
    effective_volume = torch.zeros(frames + 1, num_samples, device=device)
    predictions = torch.zeros(frames + 1, num_samples, device=device)

    # get model
    # select non linearity
    if args.nl == "Sigmoid":
        nl = nn.Sigmoid()
    elif args.nl == "Erf":
        nl = lambda x: torch.erf(x / (2 ** (1 / 2)))
    else:
        raise NotImplementedError(f"nl {args.nl} not supported")
    model = MLPSingle(
        width=w, output_dim=len(torch.unique(Y)), input_dim=X.shape[1], nl=nl
    ).to(device)

    # init models
    weights_init(model)
    loss_func = nn.CrossEntropyLoss()
    if isinstance(loss_func, nn.CrossEntropyLoss):
        Y = Y.long()  # cast to long

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
    steps = X_train.size(0)
    bs = min(args.batchsize, steps)  # batchsize
    cnt = 0
    weightlist = []
    biaslist = []

    # training
    for i in range(epochs):
        model.train()
        # specify permutations
        permute_idx = torch.randperm(X_train.shape[0], device=device).long()
        # drop last
        loops = steps // bs
        num_correct = 0
        for j in range(loops):
            data_point = permute_idx[j * bs : (j + 1) * bs]
            x_var = X_train[data_point]
            y_var = Y_train[data_point]

            optimizer.zero_grad()
            y_hat = model(x_var)
            loss = loss_func(y_hat, y_var)
            loss.backward()
            optimizer.step()

            # accuracy
            with torch.no_grad():
                num_correct += (torch.argmax(y_hat, dim=1) == y_var).float().sum()

        if i % args.print_freq == 0:
            # get test accuracy
            with torch.no_grad():
                y_hat_test = model(X_test)
                test_acc = (torch.argmax(y_hat_test, dim=1) == Y_test).float().mean()

            print(
                "Epoch: {:03}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}".format(
                    i,
                    loss.item(),
                    num_correct.item() / X_train.shape[0],
                    test_acc.item(),
                )
            )

        if i % args.save_freq == 0:
            model.eval()
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
                        model.feature_map,
                        k=args.k,
                        thr=args.thr,
                    ).squeeze()

                # put back
                effective_volume[cnt] = detarr
                cur_prediction = torch.argmax(model(scan), dim=1)  # predictions
                predictions[cnt] = cur_prediction
                # get model params
                weightlist.append(list(model.parameters())[0].clone())
                biaslist.append(list(model.parameters())[1].clone())
                cnt += 1

    # save geometric evaluations
    torch.save(model.to("cpu"), os.path.join(model_dir, "model.pt"))
    torch.save(effective_volume.to("cpu"), os.path.join(model_dir, "eff_vol.pt"))
    torch.save(predictions.to("cpu"), os.path.join(model_dir, "predictions.pt"))

    # save model parameters
    torch.save(weightlist, os.path.join(model_dir, "weightlist.pt"))
    torch.save(biaslist, os.path.join(model_dir, "biaslist.pt"))


if __name__ == "__main__":
    main()
