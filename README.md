### Response to reviewer 65Ak

> **W1**: It is suggested to show the running time of applying SnoHv 1&2 and compare it with baseline model.


To better showcase the results, we conducted tests on the medium dataset, Arxiv, using a 32-layer GNN backbone (hidden layer=64, epoch=500, V100 support). The results are as follows. And we will try our best to add more results to the revised manuscript in the future.

**Table 1.** Running time of different baseline using SnoHv2 on ogbn-arxiv. For each backbone we trained 500 epochs
| baseline (8-layer)| original  | +SnoHv2  |
|----------|-----------|----------|
| GCN      | 404.67s   | 261.44s  |
| ResGCN   | 415.49s   | 223.53s  |
| ResGCN+  | 415.41s   | 339.49s  |


We demonstrate the training time of the model after pruning. We emphasize that SnoHv1/v2 functions as an **inference accelerator**, simultaneously offering interpretability to the pruning process, rather than focusing solely on the training process. This aligns with mainstream pruning literatures, which focus on the aspects of storage and inference [1-5]. In fact, aggregation operations through adjacency matrices are a time-consuming part of graph neural networks. Therefore, the running time of our method (whether it is inference time or training time) depends on the sparsity of the adjacency matrix after pruning. The sparser the adjacency matrix, the shorter the time spent, while the original adjacency matrix is the densest. After adding SnoHv2, the baseline FLOPS will also decrease a bit, because the number of edges will decrease and the corresponding calculation will decrease.

[1] A unified lottery ticket hypothesis for graph neural networks

[2] Searching lottery tickets in graph neural networks: A dual perspective

[3] Brave the wind and the waves: Discovering robust and generalizable graph lottery tickets

[4] Comprehensive graph gradual pruning for sparse training in graph neural networks

[5] IGRP: Iterative Gradient Rank Pruning for Finding Graph Lottery Ticket































