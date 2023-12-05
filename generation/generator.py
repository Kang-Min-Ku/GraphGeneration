from loader import load_dataset, load_sample_dataset, save_graph, load_graph, label_comparison
from cal_similarity import cal_cosine_similarity, cal_euclidean_similarity
from visualization import visualization
import torch

import sys, os
utils_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils')
sys.path.append(utils_path)
from parser import YamlParser

import networkx as nx

Yamlparser = YamlParser('../hyperparam/base.yaml')
config = Yamlparser.args

dataset = config.dataset
embedding_size = config.embedding_size
similarity_method = config.similarity_method
model = config.model
edge_threshold = config.edge_threshold
sample_size = config.sample_size
batch_size = config.batch_size

file_path = '../save/' + dataset + '_' + model + '_emb_' + str(embedding_size) + '.pt'
save_path = '../graph/' + dataset + '_' + model + '_emb_' + str(embedding_size) + '_' + similarity_method + '_' + str(edge_threshold) + '_' + str(sample_size) + '.json'

# Load Dataset
# node_embeddings, labels = load_dataset(file_path)
node_embeddings, labels = load_sample_dataset(file_path, sample_size)

# Calculate Similarity
if similarity_method == 'cosine':
    cal_similarity = cal_cosine_similarity(node_embeddings)
elif similarity_method == 'euclidean':
    cal_similarity = cal_euclidean_similarity(node_embeddings)

# Save Graph
save_graph(node_embeddings, cal_similarity, labels, edge_threshold, save_path)


"""
# Label comparison
G = load_graph(save_path)
print(G.number_of_nodes())
print(G.number_of_edges())

label_comparison(G)

# visualization(G)
"""