
import argparse
import logging
import os
import random
import torch
import torch.nn as nn
import numpy as np
from utils import initialize_experiment
from data_utils import get_dataset
from model import *
from trainer import Trainer
def main(params):

    G = get_dataset(params).to(params.device)

    if params.model == 'GCN':
        graph_classifier = GCN(params).to(params.device)
    elif params.model == 'GAT':
        graph_classifier = GAT(params).to(params.device)
    elif params.model == 'MixHop':
        graph_classifier = MixHop(params).to(params.device)
    elif params.model == 'GCNJK':
        graph_classifier = GCNJK(params).to(params.device)
    elif params.model == 'H2GCN':
        graph_classifier = H2GCN(params, edge_index = G.edge_index).to(params.device)
    elif params.model == 'MLP':
        graph_classifier = MLP(in_channels = params.inp_dim, hidden_channels = params.hidden_dim, out_channels = params.out_dim, 
                               num_layers = params.num_layers, dropout = params.dropout).to(params.device)
    else:
        raise NotImplementedError
    logging.info(f"Device: {params.device}")
    trainer = Trainer(params = params, graph_classifier = graph_classifier, data = G)

    logging.info('Starting training with full batch...')
    
    trainer.train()
    
    


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='node classification model')

    # Experiment setup params
    parser.add_argument("--experiment_name", "-e", type=str, default="default",
                        help="A folder with this name would be created to dump saved models and log files")
    parser.add_argument("--dataset", "-d", type=str,
                        help="Dataset string")
    parser.add_argument("--gpu", type=int, default=0,
                        help="Which GPU to use?")

    parser.add_argument('--disable_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--model', type = str,
                        help='model_name')
    parser.add_argument("--hidden_dim", type=int, default=32,
                        help="hidden dim?")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="number of layers")
    parser.add_argument("--num_epochs", "-ne", type=int, default=2000,
                        help="number of epcohs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate of the optimizer")
    parser.add_argument("--l2", type=float, default=5e-3,
                        help="Regularization constant for GNN weights")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout rate in GNN layers")
    parser.add_argument("--train_ratio", type=float, default=0.5,
                        help="train ratio")
    parser.add_argument("--test_ratio", type=float, default=0.25,
                        help="test ratio")
    parser.add_argument("--early_stop", type=int, default=100,
                        help="Early stopping patience")
    parser.add_argument('--seed', dest='seed', default=0,
                        type=int, help='Seed for randomization')
    parser.add_argument('--hop', type=int, default=2, help='hop')
    parser.add_argument('--threshold_dog', action='store_true',
                        help='use threshold dataset')
    parser.add_argument('--threshold_cifar', action='store_true',
                        help='use threshold cifar dataset')
    params = parser.parse_args()
    params.experiment_name = f'{params.model}_{params.dataset}'

    initialize_experiment(params, __file__)

    torch.manual_seed(params.seed)
    torch.cuda.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)
    np.random.seed(params.seed)
    random.seed(params.seed)

    if not params.disable_cuda and torch.cuda.is_available():
        params.device = torch.device('cuda:%d' % params.gpu)
    else:
        params.device = torch.device('cpu')


    main(params)
