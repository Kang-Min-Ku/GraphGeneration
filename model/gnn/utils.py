from torch_geometric.transforms  import RandomNodeSplit
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data


import os
import logging
import json
import torch
import networkx as nx
import numpy as np

def nx_to_pyg(nx_graph):
    edge_index = torch.tensor(list(nx_graph.edges)).t().contiguous()
    x = torch.tensor([nx_graph.nodes[i]['embedding'] for i in nx_graph.nodes], dtype=torch.float)
    y = torch.tensor([nx_graph.nodes[i]['label'] for i in nx_graph.nodes], dtype=torch.long)
    edge_attr = torch.tensor([nx_graph.edges[i, j]['weight'] for i, j in nx_graph.edges], dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)
    return data
def initialize_experiment(params, file_name):
    '''
    Makes the experiment directory, sets standard paths and initializes the logger
    '''
    params.main_dir = os.path.join(os.path.relpath(os.path.dirname(os.path.abspath(__file__))))
    exps_dir = os.path.join(params.main_dir, 'experiments')
    print(exps_dir)
    if not os.path.exists(exps_dir):
        os.makedirs(exps_dir)

    params.exp_dir = os.path.join(exps_dir, params.experiment_name)
    
    if not os.path.exists(params.exp_dir):
        os.makedirs(params.exp_dir)

    file_handler = logging.FileHandler(os.path.join(params.exp_dir, "log_train.txt"))

    logger = logging.getLogger()
    logger.addHandler(file_handler)

    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v
                          in sorted(dict(vars(params)).items())))
    logger.info('============================================')

    with open(os.path.join(params.exp_dir, "params.json"), 'w') as fout:
        json.dump(vars(params), fout)
        
        
