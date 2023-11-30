from loader import load_dataset, save_graph
from cal_similarity import cal_cosine_similarity, cal_euclidean_similarity
import torch

import sys, os
utils_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils')
sys.path.append(utils_path)
from parser import YamlParser

Yamlparser = YamlParser('../hyperparam/base.yaml')
config = Yamlparser.args.config

dataset = config['dataset']
embedding_size = config['embedding_size']
similarity_method = config['similarity_method']
model = config['model']
edge_threshold = config['edge_threshold']

file_path = '../data/' + dataset + '_' + model + '_emb_' + str(embedding_size) + '.pt'
save_path = '../graph/' + dataset + '_' + model + '_emb_' + str(embedding_size) + '_' + similarity_method + '_' + str(edge_threshold) + '.gml'

# Load Dataset
# embeddings, labels = load_dataset(file_path)
embeddings, labels = load_dataset('../data/small_dummy.pt')

# Calculate Similarity
if similarity_method == 'cosine':
    cal_similarity = cal_cosine_similarity(embeddings)
elif similarity_method == 'euclidean':
    cal_similarity = cal_euclidean_similarity(embeddings)

# Save Graph
save_graph(cal_similarity, labels, edge_threshold, file_path)