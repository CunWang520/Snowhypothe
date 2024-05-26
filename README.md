# SnoHv2
For cora,citeseer,pubmed datasets, we trained in a fully supervised form
For ogbn-arxiv, we use the official split
## Default 
    --dataset cora
    --dspath ./data
    --seed 42
    --device 0
    --epochs 500
    --lr 0.01
    --weight_decay  5e-4
    --block gcn
    --numlayer 32
    --hidden_channels 64
    --withbn False
    --stop_rate 0.1
## Usage
### gcn 16layer(without batch norm)
	python main.py --dataset cora/citeseer/pubmed --epochs 500 --lr 0.001 --block gcn --numlayer 16 --stop_rate 0.01
### gcn 32layer(+batch norm)
    python main.py --dataset cora/citeseer/pubmed --epochs 500 --lr 0.01 --block gcn --numlayer 32 --stop_rate 0.2 --withbn
    python main.py --dataset ogbn-arxiv --epochs 500 --lr 0.01 --block gcn --numlayer 32 --stop_rate 0.05 --withbn
    
### gcn 64layer(+batch norm)
    python main.py --dataset cora/citeseer/pubmed --epochs 1000 --lr 0.01 --block gcn --numlayer 64 --stop_rate 0.01 --withbn
    python main.py --dataset ogbn-arxiv --epochs 1000 --lr 0.01 --block gcn --numlayer 64 --stop_rate 0.01 --withbn

### resgcn 32layer
    cora/citeseer:
        python main.py --dataset cora/citeseer --epochs 500 --lr 0.01 --block resgcn --numlayer 32 --stop_rate 0.1 --withbn
    pubmed:
        python main.py --dataset pubmed --epochs 500 --lr 0.01 --block resgcn --numlayer 32 --stop_rate 0.15 --withbn
    ogbn-arxiv: 
        python main.py --dataset ogbn-arxiv --epochs 500 --lr 0.01 --block resgcn --numlayer 32 --stop_rate 0.1 --withbn
### resgcn 64layer
    python main.py --dataset cora/citeseer/pubmed --epochs 1000 --lr 0.01 --block resgcn --numlayer 64 --stop_rate 0.1 --withbn
    python main.py --dataset ogbn-arxiv --epochs 1000 --lr 0.01 --block resgcn --numlayer 32 --stop_rate 0.02 --withbn

### resgcn+ 32layer
    cora/citeseer:
        python main.py --dataset cora/citeseer --epochs 500 --lr 0.01 --block resgcn+ --numlayer 32 --stop_rate 0.4 --withbn
    pubmed:
        python main.py --dataset pubmed --epochs 500 --lr 0.01 --block resgcn+ --numlayer 32 --stop_rate 0.5 --withbn
    ogbn-arxiv:
        python main.py --dataset ogbn-arxiv --epochs 500 --lr 0.01 --block resgcn+ --numlayer 32 --stop_rate 0.05 --withbn
### resgcn+ 64layer
    python main.py --dataset cora/citeseer/pubmed --epochs 1000 --lr 0.01 --block resgcn+ --numlayer 64 --stop_rate 0.7 --withbn
    python main.py --dataset ogbn-arxiv --epochs 500 --lr 0.01 --block resgcn+ --numlayer 64 --stop_rate 0.02 --withbn

![fig2](EXPALAIN.jpg)





