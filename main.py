import copy
import os.path
import time

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Sequential
import torch.nn.functional as F
import random
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from torch_geometric.nn import GCNConv
# from SnoHModels import SnoHGCNLayer as GCNConv
from torch.nn import BatchNorm1d
import torch.nn as nn
import torch
import torch.nn.functional as F

import utils
from SnoHModels import SnoHGCN
from utils import get_layers_adj
from load_dataset import load_planetoid, load_arxiv
from torch_geometric.nn.models import GCN
from torch.optim.lr_scheduler import MultiStepLR
import argparse


def train_get_edge_index(model, x, edge_index, labels, optimizer, split_idx,  total_epochs, perf_save_path, filename):
    print("="*50+"train origin gnn model"+"="*50)
    if not os.path.exists(perf_save_path):
        os.makedirs(perf_save_path)

    train_idx = split_idx["train"]
    val_idx = split_idx["valid"]
    test_idx = split_idx["test"]

    edge_indexes = [edge_index] * model.numlayer
    model.get_edge_indexes(edge_indexes)

    acc_test = np.zeros(total_epochs)
    acc_train = np.zeros(total_epochs)
    acc_val = np.zeros(total_epochs)

    best_acc_dist = []
    best_val_acc = 0.0
    for epoch in range(total_epochs):
        torch.cuda.empty_cache()
        model.train()
        optimizer.zero_grad()
        out, layers_cos_dist = model(x)

        loss = F.nll_loss(out[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            pred, _ = model(x)
            pred = pred.argmax(dim=1)
            val_correct = (pred[val_idx] == labels[val_idx]).sum()
            val_acc = int(val_correct) / len(val_idx)
            acc_val[epoch] = val_acc

            train_correct = (pred[train_idx] == labels[train_idx]).sum()
            train_acc = int(train_correct) / len(train_idx)
            acc_train[epoch] = train_acc

            test_correct = (pred[test_idx] == labels[test_idx]).sum()
            test_acc = int(test_correct) / len(test_idx)
            acc_test[epoch] = test_acc

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_acc_dist = layers_cos_dist
                test_under_best_val = test_acc
            print(f'epoch: {epoch}, train loss {loss.item():.4f}, Accuracy: {test_acc:.4f}, || best val acc: {best_val_acc:.4f} test: {test_under_best_val:.4f}')
    np.save(os.path.join(perf_save_path, filename+"_train.npy"), acc_train)
    np.save(os.path.join(perf_save_path, filename+"_val.npy"), acc_val)
    np.save(os.path.join(perf_save_path, filename+"_test.npy"), acc_test)
    return best_acc_dist


def train_fix_edge_index(model, x, labels, optimizer, split_idx, total_epochs, perf_save_path, filename):
    print("=" * 50 + "train snoh gnn model" + "=" * 50)
    if not os.path.exists(perf_save_path):
        os.makedirs(perf_save_path)

    train_idx = split_idx["train"]
    val_idx = split_idx["valid"]
    test_idx = split_idx["test"]

    acc_test = np.zeros(total_epochs)
    acc_train = np.zeros(total_epochs)
    acc_val = np.zeros(total_epochs)

    best_val_acc = 0.0
    for epoch in range(total_epochs):
        torch.cuda.empty_cache()
        model.train()
        optimizer.zero_grad()
        out, layers_cos_dist = model(x)
        loss = F.nll_loss(out[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()

            pred, _ = model(x)
            pred = pred.argmax(dim=1)
            val_correct = (pred[val_idx] == labels[val_idx]).sum()
            val_acc = int(val_correct) / len(val_idx)
            acc_val[epoch] = val_acc

            train_correct = (pred[train_idx] == labels[train_idx]).sum()
            train_acc = int(train_correct) / len(train_idx)
            acc_train[epoch] = train_acc

            test_correct = (pred[test_idx] == labels[test_idx]).sum()
            test_acc = int(test_correct) / len(test_idx)
            acc_test[epoch] = test_acc
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_acc_dist = layers_cos_dist
                test_under_best_val = test_acc
            print(
                f'epoch: {epoch}, train loss {loss.item():.4f}, Accuracy: {test_acc:.4f}, || best val acc: {best_val_acc} test: {test_under_best_val}')
    np.save(os.path.join(perf_save_path, filename+"_train.npy"), acc_train)
    np.save(os.path.join(perf_save_path, filename+"_val.npy"), acc_val)
    np.save(os.path.join(perf_save_path, filename+"_test.npy"), acc_test)
    return best_acc_dist


def main():
    parser = argparse.ArgumentParser(description='SnoHv2')
    # dataset
    parser.add_argument('--dataset', type=str, default='cora', choices=["ogbn-arxiv", "cora", "citeseer", "pubmed"],
                        help='dataset name (default: ogbn-arxiv)')
    parser.add_argument('--dspath', type=str, default='./data', help="dataset path")
    # train
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train (default: 500)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate set for optimizer.')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    # parser.add_argument('--dropout', type=float, default=0.)
    # model
    parser.add_argument('--block', type=str, default="resgcn+", choices=["gcn", "resgcn", "resgcn+"])
    parser.add_argument('--numlayer', type=int, default=32,
                        help='the number of layers of the networks')
    parser.add_argument('--hidden_channels', type=int, default=64,
                        help='the dimension of embeddings of nodes')
    parser.add_argument('--withbn', action="store_true")
    # save settings
    parser.add_argument('--save_model', action="store_true")  # TODO: save model and edge_index
    parser.add_argument('--model_save_path', type=str, default='model_ckpt',
                        help="the directory used to save models")
    parser.add_argument('--exp_name', type=str, default='EXP', help="experiment name")
    # prune
    parser.add_argument("--stop_rate", type=float, default=0.1, help="early stopping parameter of node")
    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else "cpu"

    if args.dataset.lower() in ["cora", "citeseer", "pubmed"]:
        dataset, graph, num_features, num_classes, split_idx, evaluator = load_planetoid(args.dataset, args.dspath)
    else:
        dataset, graph, num_features, num_classes, split_idx, evaluator = load_arxiv(args.dspath)

    start_time = time.strftime("%Y%m%d-%H%M%S")
    exp_name = f"{args.exp_name}_data-{args.dataset}_lr-{args.lr}_wd-{args.weight_decay}_block-{args.block}_numlayer-{args.numlayer}_{start_time}"
    if args.withbn:
        exp_name = exp_name + "_withbn"
    perf_save_path = os.path.join("./experment", exp_name)

    x, edge_index = graph.x.to(device), graph.edge_index.to(device)
    labels = graph.y.to(device)

    model = SnoHGCN(in_channels=dataset.num_node_features,
                    mid_channels=args.hidden_channels,
                    out_channels=dataset.num_classes,
                    numlayer=args.numlayer,
                    activation=F.relu,
                    layerwithbn=args.withbn,
                    block=args.block)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model_fix_node = copy.deepcopy(model)
    optimizer_fix_node = torch.optim.Adam(model_fix_node.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model = model.to(device)
    print(model)
    layers_cos_dist = train_get_edge_index(model, x, edge_index, labels, optimizer, split_idx,
                                           total_epochs=args.epochs, perf_save_path=perf_save_path, filename="base_accuracy")
    layers_edge_index, layers_node_num, _, _, thre_layer = utils.get_layers_adj(layers_cos_dist, args.stop_rate, edge_index.cpu())
    utils.print_sparsity(layers_node_num, layers_edge_index)

    for i in range(len(layers_edge_index)):
        layers_edge_index[i] = layers_edge_index[i].to(device)
    model_fix_node.edge_indexs = model_fix_node.get_edge_indexes(layers_edge_index)
    model_fix_node = model_fix_node.to(device)
    train_fix_edge_index(model_fix_node, x, labels, optimizer_fix_node, split_idx,
                         total_epochs=args.epochs, perf_save_path=perf_save_path, filename="snoh_accuracy")


if __name__ == "__main__":
    main()