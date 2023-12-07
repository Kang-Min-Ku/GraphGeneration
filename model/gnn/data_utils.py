from utils import nx_to_pyg
from torch_geometric.data import Data
from torch_geometric.transforms  import RandomNodeSplit
import torch
import networkx as nx
import json
import numpy as np
import random
        
def get_dataset(params):
    if params.dataset == 'Cora':
        cora_graph = Planetoid(root='.', name='Cora')
        cora_graph.data.x = torch.randn(cora_graph.data.num_nodes, 32)
        cora_graph.data.edge_attr = torch.randn(cora_graph.data.edge_index.shape[1])
        g_nx = nx.Graph()

        for i in range(cora_graph.data.num_nodes):
            g_nx.add_node(i, embedding=cora_graph.data.x[i].tolist(), label=int(cora_graph.data.y[i]))

        edge_index = cora_graph.data.edge_index.numpy()
        edge_attr = cora_graph.data.edge_attr.numpy()
        for i in range(edge_index.shape[1]):
            g_nx.add_edge(edge_index[0, i], edge_index[1, i], attr=edge_attr[i])


    else:
        if not params.threshold_dog and not params.threshold_cifar:
            file_path = f"data/{params.dataset}.json"
        elif params.threshold_dog and not params.threshold_cifar:
            file_path = f"data/threshold_dog/{params.dataset}.json"
        elif params.threshold_cifar and not params.threshold_dog:
            file_path = f"data/threshold_cifar/{params.dataset}.json"
        else:
            raise Exception("Dataset error")
        with open(file_path, "r") as json_file:
            g_nx = nx.node_link_graph(json.load(json_file))
    G = nx_to_pyg(g_nx)
    if (params.dataset.split('_')[0] == 'cifar100' or params.dataset.split('_')[1] == 'cifar100') and params.dataset.split('_')[-1] == '60000':

        train_mask = torch.zeros(60000, dtype = bool)
        test_mask = torch.zeros(60000, dtype = bool)
        test_mask[50000:] = True
        G.test_mask = test_mask
        train_idx = random.sample(list(range(50000)), 40000)
        train_mask[train_idx] = True
        G.train_mask = train_mask
        G.val_mask = torch.logical_not(G.train_mask)
        G.val_mask[50000:] = False
        

    elif (params.dataset.split('_')[0] == 'dog' or params.dataset.split('_')[1] == 'dog') and params.dataset.split('_')[-1] == '20580':
        train_mask = torch.zeros(20580, dtype = bool)
        test_mask = torch.zeros(20580, dtype = bool)
        test_mask[16464:] = True
        G.test_mask = test_mask
        train_idx = random.sample(list(range(16464)), int(16464 * 0.75))
        train_mask[train_idx] = True
        G.train_mask = train_mask
        G.val_mask = torch.logical_not(G.train_mask)
        G.val_mask[16464:] = False

    else:
        num_data = G.x.shape[0]
        num_train = int(params.train_ratio * num_data)
        num_test = int(params.test_ratio * num_data)
        num_val = num_data - num_train - num_test

        G = RandomNodeSplit(num_val = num_val, num_test = num_test)(G)
    params.inp_dim = G.x.shape[1]
    params.num_nodes = G.x.shape[0]
    params.out_dim = len(set(G.y.tolist()))

    return G

