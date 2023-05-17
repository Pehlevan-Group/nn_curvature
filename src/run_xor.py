"""
animations of curvature growth over training of variable width single hidden-layer XOR network
"""

# load packages
import os
import argparse
import warnings

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# load file
from data import XORdata, noisyXORdata, sindata, sindata_full, noisy_sindata_train_test
from model import MLPSingle, weights_init
from utils import (
    scalcurvfromric,
    expansion,
    load_config,
    fileid,
    ricci_analytic,
    determinant_analytic,
)

# arguments
parser = parser = argparse.ArgumentParser()

# model params
parser.add_argument(
    "--data",
    default="XOR",
    type=str,
    help="data tested",
    choices=["XOR", "noisyXOR", "sindata", "sindata_full", "noisy_sindata_train_test"],
)
parser.add_argument(
    "--test-size",
    default=0.2, type=float,
    help='the proportion of testing dataset'
)
parser.add_argument(
    "--std",
    default=0.5, type=float,
    help='the standard deviation for Gaussian perturbation in the training set'
)
parser.add_argument("--w", default=2, type=int, help="tested widths")
parser.add_argument("--output-dim", default=1, type=int, help="the output dimension")
parser.add_argument(
    "--nl", default="Sigmoid", type=str, help="the type of nonlinearity"
)
parser.add_argument("--epochs", default=2000, type=int, help="the number of epochs")
parser.add_argument(
    "--batchsize",
    default=50,
    type=int,
    help="size of batch; if exceeding datasize, fullbatch is used",
)
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
    "--scanbatchsize",
    default=40,
    type=int,
    help="the number of scan points for geometric quanities computations",
)
parser.add_argument(
    "--use-analytic",
    default=False,
    action="store_true",
    help="true to use analytic formula for volume element and curvature computations",
)

# for adding volume elements as a loss term 
parser.add_argument(
    "--no-geometric-quantities",
    action='store_true', 
    default=False,
    help='turn on to skip logging geoemtric quantities'
)
parser.add_argument(
    "--burnin",
    default=2000, 
    type=int, 
    help='the burn in period to add volume element as an optimizer'
)
parser.add_argument(
    "--_lambda",
    default=100,
    type=float,
    help='the multiplier for the loss on the volume element term '
)
parser.add_argument('--ent-thr', default=0.5, type=float, help='the cutoff above which points are negelected')

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

    l = torch.linspace(args.lower, args.upper, steps=args.steps).to(device)
    y = torch.linspace(args.lower, args.upper, steps=args.steps).to(device)

    scan = torch.cartesian_prod(l, y)
    num_samples = args.steps**2
    frames = args.epochs // args.save_freq

    weightlist = []
    biaslist = []

    # select data
    if args.data == "XOR":
        X, Y = XORdata()
    elif args.data == "noisyXOR":
        X, Y = noisyXORdata()
    elif args.data == "sindata":
        X, Y = sindata(seed=args.seed)
    elif args.data == "sindata_full":
        X, Y = sindata_full()
    elif args.data == 'noisy_sindata_train_test':
        X, X_test, Y, Y_test = noisy_sindata_train_test(seed=args.seed, test_size=args.test_size, std=args.std)
        X_test = X_test.to(device)
        Y_test = Y_test.to(device)
        train_test_specifier = f'_std{args.std}_bi{args.burnin}_l{args._lambda}'
    else:
        raise NotImplementedError(f"data {args.data} not supported")
    X, Y = X.to(device), Y.to(device)

    # run for each w
    w = args.w
    # store metrics
    colorchange = torch.zeros(frames + 1, num_samples, device=device)
    curvtens = torch.zeros(frames + 1, num_samples, device=device)
    dettens = torch.zeros(frames + 1, num_samples, device=device)

    # get model
    # select non linearity
    if args.nl == "Sigmoid":
        nl = nn.Sigmoid()
    elif args.nl == "Erf":
        nl = lambda x: torch.erf(x / (2 ** (1 / 2)))
    elif args.nl == "ReLU":
        nl = nn.ReLU()
        warnings.warn("Caution: ReLU is not smooth")
    else:
        raise NotImplementedError(f"nl {args.nl} not supported")
    model = MLPSingle(width=w, output_dim=args.output_dim, nl=nl).to(device)

    # setup model paths and result paths
    model_id = fileid("xor", args)
    if 'train_test' in args.data: model_id += train_test_specifier
    model_dir = os.path.join(paths["model_dir"], model_id)
    result_dir = os.path.join(paths["result_dir"], model_id)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # init tb 
    writer = SummaryWriter(result_dir)

    # init models
    weights_init(model)
    loss_func = getattr(nn, args.loss)()
    task_type = 'regression'  # annotate current task type
    if isinstance(loss_func, nn.CrossEntropyLoss):
        Y = Y.long()  # cast to long
        softmax = nn.Softmax(dim=-1)
        task_type = 'classification'
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    epochs = args.epochs + 1
    steps = X.size(0)
    bs = min(args.batchsize, steps)  # batchsize
    cnt = 0
    weightlist = []
    biaslist = []

    # training
    for i in range(epochs):
        model.train()
        # specify permutations
        permute_idx = torch.randperm(X.shape[0], device=device).long()
        # drop last
        loops = steps // bs
        for j in range(loops):
            data_point = permute_idx[j * bs : (j + 1) * bs]
            x_var = X[data_point]
            y_var = Y[data_point]

            optimizer.zero_grad()
            y_hat = model(x_var)
            loss = loss_func(y_hat.squeeze(), y_var.squeeze())

            # optimize geometric quantites 
            if i >= args.burnin:
                y_hat_probs = softmax(y_hat)
                entropies = - (y_hat_probs * y_hat_probs.log2()).sum(dim=-1)  # TODO: log2?
                candidate_indices = torch.nonzero(entropies >= args.ent_thr).flatten()
                num_candidates = candidate_indices.shape[0]
                
                # feed in by batch due to memory constraints
                num_scan_loops = int(np.ceil(num_candidates / args.scanbatchsize))

                geometric_loss = 0
                for loop in range(num_scan_loops):
                    start_idx = loop * args.scanbatchsize
                    end_idx = (loop + 1) * args.scanbatchsize
                    # analytic scalar curvature computations
                    if args.use_analytic:
                        # get linear layer parameters
                        W, b = model.lin1.parameters()
                        geometric_loss += determinant_analytic(
                            X[candidate_indices[start_idx:end_idx]], W, b, args.nl
                        ).sum()
                    # autograd
                    else:
                        geometric_loss += expansion(
                            X[candidate_indices[start_idx:end_idx]],
                            model.feature_map,
                        ).sum()

                loss -= args._lambda * geometric_loss
            
            loss.backward()
            optimizer.step()

        if i % args.print_freq == 0:
            writer.add_scalar('Train/Loss', loss, i)

            if task_type == 'classification':
                with torch.no_grad():
                    # get accuracy 
                    acc = (y_hat.argmax(dim=-1) == y_var.squeeze()).float().mean()
                    writer.add_scalar('Train/Acc', acc, i)
                    if 'train_test' in args.data: 
                        # evaluate test 
                        model.eval()
                        test_pred = model(X_test).argmax(dim=-1)
                        test_acc = (test_pred == Y_test.squeeze()).float().mean()
                        writer.add_scalar("Test/Acc", test_acc, i)
                        print("Epoch {:03}, Loss: {:.6f}, Train Acc: {:.4f}, Test Acc: {:.4f}".format(i, loss.item(), acc.item(), test_acc.item()))
                    else:
                        print("Epoch {:03}, Loss: {:.6f}, Acc: {:.4f}".format(i, loss.item(), acc.item()))
            else:
                print("Epoch: {:03}, Loss: {:.6f}".format(i, loss.item()))

            writer.flush()

        if i % args.save_freq == 0 and not args.no_geometric_quantities:
            model.eval()
            with torch.no_grad():
                # get geometric quantities
                scalarr = torch.zeros((scan.size()[0])).to(device)
                detarr = torch.zeros((scan.size()[0])).to(device)

                # feed to computation by batch
                num_scan_loops = int(np.ceil(scan.size()[0] / args.scanbatchsize))
                for loop in range(num_scan_loops):
                    start_idx = loop * args.scanbatchsize
                    end_idx = (loop + 1) * args.scanbatchsize
                    # analytic scalar curvature computations
                    if args.use_analytic:
                        # get linear layer parameters
                        W, b = model.lin1.parameters()
                        scalarr[start_idx:end_idx] = ricci_analytic(
                            scan[start_idx:end_idx], W, b, args.nl
                        ).squeeze()
                        detarr[start_idx:end_idx] = determinant_analytic(
                            scan[start_idx:end_idx], W, b, args.nl
                        ).squeeze()
                    # autograd
                    else:
                        scalarr[start_idx:end_idx] = scalcurvfromric(
                            scan[start_idx:end_idx],
                            model.feature_map,
                        ).squeeze()
                        detarr[start_idx:end_idx] = expansion(
                            scan[start_idx:end_idx],
                            model.feature_map,
                        ).squeeze()

                # put back
                curvtens[cnt] = scalarr
                dettens[cnt] = detarr
                if isinstance(loss_func, nn.MSELoss):
                    colorchange[cnt] = model(scan).squeeze()
                elif isinstance(loss_func, nn.CrossEntropyLoss):
                    colorchange[cnt] = torch.argmax(model(scan), dim=1)

                # get model params
                weightlist.append(list(model.parameters())[0].clone())
                biaslist.append(list(model.parameters())[1].clone())
                cnt += 1

    # save geometric evaluations
    if not args.no_geometric_quantities:
        torch.save(model.to("cpu"), os.path.join(model_dir, "model.pt"))
        torch.save(dettens.to("cpu"), os.path.join(model_dir, "dettens.pt"))
        torch.save(curvtens.to("cpu"), os.path.join(model_dir, "curvtens.pt"))
        torch.save(colorchange.to("cpu"), os.path.join(model_dir, "colorchange.pt"))

        # save model parameters
        torch.save(weightlist, os.path.join(model_dir, "weightlist.pt"))
        torch.save(biaslist, os.path.join(model_dir, "biaslist.pt"))


if __name__ == "__main__":
    main()
