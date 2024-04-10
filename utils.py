import torch
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt


def random_sample_edge(edge_indexes:list, layernum, percent):
    if len(edge_indexes) < 1:
        raise Exception("len(edge_indexes) must be greater than or equal to 1")
    device = edge_indexes[0].device
    new_edge_indexes = []
    if len(edge_indexes) == 1:
        edge_index = edge_indexes[0]
        edge_num = edge_index.shape[1]
        perserve_edgenum = int(edge_num * percent)
        perm = torch.randperm(edge_num).to(device)
        perm = perm[:perserve_edgenum]
        edge_index = edge_index[:, perm]
        new_edge_indexes.append(edge_index)
        new_edge_indexes = new_edge_indexes * layernum
    else:
        assert len(edge_indexes) == layernum
        for i in range(len(edge_indexes)):
            edge_index = edge_indexes[i]
            edge_num = edge_index.shape[1]
            perserve_edgenum = int(edge_num * percent)
            perm = torch.randperm(edge_num).to(device)
            perm = perm[:perserve_edgenum]
            edge_index = edge_index[:, perm]
            new_edge_indexes.append(edge_index)

    return new_edge_indexes


def get_layers_adj(layers_cos_dist, stopping_rate, edge_index):
    for i in range(len(layers_cos_dist)):
        layers_cos_dist[i][layers_cos_dist[i] < 0] = 0
    if len(layers_cos_dist) == 0 or len(layers_cos_dist) == 1:
        raise ValueError("layers_cos_dist can not be 0 and 1, the layers num of network must be greater than 2")
    layers_graph_dist = []
    for i in range(len(layers_cos_dist)):
        layers_graph_dist.append(layers_cos_dist[i].sum())

    # the first layer may not be the max diff
    thre_layer = 0
    for i in range(1, len(layers_graph_dist)):
        if layers_graph_dist[i] > layers_graph_dist[thre_layer]:
            thre_layer = i
        else:
            break

    layers_node_mask = []
    layers_node_num = []
    for i in range(0, thre_layer):
        layers_node_mask.append(torch.ones_like(layers_cos_dist[0], dtype=torch.bool))
        layers_node_num.append(len(layers_node_mask[i]))
    thre = layers_cos_dist[thre_layer] * stopping_rate

    for i in range(thre_layer, len(layers_cos_dist)):
        i_mask = layers_cos_dist[i] >= thre
        if i != 0:
            i_mask = i_mask * layers_node_mask[i-1]
        layers_node_mask.append(i_mask)
        layers_node_num.append(i_mask.sum())

    src = edge_index[0]
    tgt = edge_index[1]

    layers_edge_index = []
    for i in range(len(layers_cos_dist)):
        edge_mask = layers_node_mask[i][tgt]
        src_new = src[edge_mask]
        tgt_new = tgt[edge_mask]
        layers_edge_index.append(torch.vstack([src_new, tgt_new]))
    return layers_edge_index, layers_node_num, layers_node_mask, layers_graph_dist, thre_layer


def display_graph_diff(layers_cos_dist, plot=False, npsavedir=None):
    if len(layers_cos_dist) == 0:
        raise ValueError("layers_cos_dict can not be zero")
    layers_graph_dist = np.array([cos_dist.sum().numpy() for cos_dist in layers_cos_dist])
    if npsavedir is not None:
        np.save(npsavedir, layers_graph_dist)
    print("graph distance: ")
    for i in range(len(layers_graph_dist)):
        cur_layer = i
        print("layer {}: {}".format(cur_layer, layers_graph_dist[cur_layer]))
    if plot:
        lgd_np = np.array(layers_graph_dist)
        plt.plot(lgd_np)
        plt.xlabel("layer")
        plt.ylabel("graph distance")
        plt.show()
    return layers_graph_dist


def print_sparsity(layers_node_num, layers_edge_index):
    if len(layers_node_num) == 0 or len(layers_edge_index) == 0:
        raise ValueError("layers_node_num or layers_edge_index can not be zero")

    # layers = len(layers_node_num)
    total_node = layers_node_num[0]
    layers_node_sparsity = []
    print("="*100)
    print("node sparsity: ")
    for i in range(len(layers_node_num)):
        cur_layer = i
        sparsity = (layers_node_num[cur_layer] / (total_node + 1e-7)) * 100
        print("layer {} node sparsity: {:.2f}%".format(cur_layer, sparsity))
        layers_node_sparsity.append(sparsity)
    print("="*30)
    print("edge sparsity: ")
    total_edge = layers_edge_index[0].shape[1]
    layers_edge_sparsity = []
    for i in range(len(layers_node_num)):
        cur_layer = i
        sparsity = (layers_edge_index[i].shape[1] / (total_edge + 1e-7)) * 100
        print("layer {} edge sparsity: {:.2f}%".format(cur_layer, sparsity))
        layers_edge_sparsity.append(sparsity)
    print("="*100)
    return layers_node_sparsity, layers_edge_sparsity