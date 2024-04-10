import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import remove_self_loops, add_self_loops, to_undirected
import torch_geometric.transforms as T
default_root_dir = "~/dataset"
import torch


def load_planetoid(dataset_name, root_dir=None, split_ratio=None, self_loops=False):
    dataset = Planetoid(root=root_dir, name=dataset_name, transform=T.NormalizeFeatures(), split="full")
    graph = dataset[0]
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    num_nodes = graph.x.size(0)

    if self_loops:
        graph.edge_index = add_self_loops(graph.edge_index)[0]
    else:
        graph.edge_index = remove_self_loops(graph.edge_index)[0]

    if split_ratio is not None:
        if np.array(split_ratio).sum() != 1:
            raise Exception("split_ratio must sum to 1")
        else:
            ids = torch.randperm(num_nodes)
            s1 = int(num_nodes * split_ratio[0])
            s2 = int(num_nodes * (split_ratio[0]+split_ratio[1]))
            split_idx = {
                "train": ids[:s1],
                "valid": ids[s1:s2],
                "test": ids[s2:]
            }
    else:
        split_idx = {
            "train": torch.nonzero(graph.train_mask).squeeze(1),
            "valid": torch.nonzero(graph.val_mask).squeeze(1),
            "test": torch.nonzero(graph.test_mask).squeeze(1),
        }

    train_num, val_num, test_num = split_idx["train"].size(0), split_idx["valid"].size(0), split_idx["test"].size(0)
    print(f"train num:{train_num}, valid num:{val_num},test num:{test_num}")
    evaluator = Evaluator(name='ogbn-arxiv')

    return dataset, graph, num_features, num_classes, split_idx, evaluator


def load_arxiv(root_dir):
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=root_dir)
    split_idx = dataset.get_idx_split()
    num_features, num_classes = dataset.num_features, dataset.num_classes
    graph = dataset[0]
    graph.y = graph.y.squeeze(-1)
    graph.edge_index = to_undirected(graph.edge_index)
    evaluator = Evaluator(name='ogbn-arxiv')
    return dataset, graph, num_features, num_classes, split_idx, evaluator
