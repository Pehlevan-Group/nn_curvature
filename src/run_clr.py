"""
run SimCLR
"""

# load packages
import os
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

# load file
from data import cifar10_contrastive
from model import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, SimCLR
from utils import load_config, fileid

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
parser.add_argument("--epochs", default=200, type=int,
                    help="the number of epochs")
parser.add_argument(
    "--batchsize",
    default=12,
    type=int,
    help="size of batch; if exceeding datasize, fullbatch is used",
)
parser.add_argument("--nl", default="GELU", type=str,
                    help="the type of nonlinearity")
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
parser.add_argument("--lr", default=0.001, type=float,
                    help="the learning rate for SGD")
parser.add_argument("--momentum", default=0.9, type=float,
                    help="the momentum in SGD")
parser.add_argument(
    "--weight-decay", default=0, type=float, help="the weight decay for SGD"
)

# technical
parser.add_argument(
    "--no-gpu", default=False, action="store_true", help="turn on to disable gpu usage"
)
parser.add_argument(
    "--seed", default=400, type=int, help="the random seed to run init and SGD"
)

# IO
parser.add_argument("--tag", default="exp", type=str,
                    help="the tag for path configs")

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
    model_id = fileid("simclr", args)
    
    model_dir = os.path.join(paths["model_dir"], model_id)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    # initialize writer
    writer = SummaryWriter(model_dir)

    # load data
    if args.data == "cifar10":
        train_dataset, unaugmented_train_dataset, test_dataset = cifar10_contrastive(paths["data_dir"])
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batchsize,
            shuffle=True,
            num_workers=2,
            drop_last=True,
        )
        # obtain the feature representation of all images
        unaugmented_train_loader = torch.utils.data.DataLoader(
            unaugmented_train_dataset,
            batch_size=args.batchsize,
            shuffle=False,  # no shuffling
            num_workers=2,
            drop_last=False,  # get all
        )
    else:
        raise NotImplementedError(f"dataset {args.data} not available")

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
    feature_map = model.feature_map
    nce_loss = model.nce_loss
    model = model.to(device)
    # if device.type == "cuda":
    #     model = nn.DataParallel(model)
    #     cudnn.benchmark = True  # reproducibility

    # init loss
    loss_func = nn.CrossEntropyLoss()

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

    # ===============================================
    # ----------- step 1: Contrastive ---------------
    # ===============================================
    print("===== Stage 1: Contrastive Learning =====")
    # training
    for i in range(epochs):
        model.train()

        train_loss = 0
        correct = 0
        total = 0
        for inputs, _ in train_loader:
            # stack multiple augmentations
            inputs = torch.cat(inputs, dim=0).to(device)
            optimizer.zero_grad()

            projections = model(inputs)
            logits, labels = nce_loss(projections)
            logits = logits / args.tempreture
            loss = loss_func(logits, labels)
            loss.backward()
            optimizer.step()


            # accuracy (of correctly identifying the positive samples)
            with torch.no_grad():
                train_loss += loss.item()
                predicted = logits.argmax(dim=1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        train_acc = correct / total

        print(
            "Epoch: {:04}, Train Loss: {:.5f}, Train Acc: {:.5f}".format(
                i, train_loss, train_acc
            )
        )

        # record
        writer.add_scalar("clr/acc", train_acc, global_step=i)
        writer.add_scalar("clr/loss", loss, global_step=i)
        writer.flush()

        # record model and features 
        if i in args.save_epochs:
            
            # snapshot of model 
            model.eval()
            torch.save(model.state_dict(), os.path.join(model_dir, f"simclr_model_state_dict_e{i}.pt"))

            print(f"save model at epoch {i}")


if __name__ == "__main__":
    main()
