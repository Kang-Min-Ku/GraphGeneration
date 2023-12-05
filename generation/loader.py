import torch
import networkx as nx
import random
import json
import numpy as np

def load_dataset(file_path):
    # .pt file loader
    embeddings, labels = torch.load(file_path)
    
    # Extract Embedding, Label 
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).clone().detach()
    labels_tensor = torch.tensor(labels, dtype=torch.int32).clone().detach()
    
    return embeddings_tensor, labels_tensor

def load_sample_dataset(file_path, sample_size):
    # .pt file loader
    embeddings, labels = torch.load(file_path)
    
    # Randomly sample a subset of data
    indices = random.sample(range(len(embeddings)), sample_size)
    sampled_embeddings = torch.stack([embeddings[i] for i in indices])  # 변환된 텐서로 샘플링된 데이터 저장
    sampled_labels = torch.tensor([labels[i] for i in indices], dtype=torch.int32)  # 라벨 텐서 생성
    
    return sampled_embeddings, sampled_labels


def save_graph(node_embeddings, similarity_matrix, labels, edge_threshold, file_path):
    # Save graph
    # Input: Node Embedding Tensor, Similarity Tensor, Label Tensor, Edge Threshold, File Path
    # Output: None
    
    G = nx.Graph()
    
    labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels  # Convert labels to NumPy if it's a PyTorch tensor
    
    for idx, node_embedding in enumerate(node_embeddings):
        embedding_list = node_embedding.tolist()
        G.add_node(idx, embedding=embedding_list, label=labels[idx].item())

    edges = threshold_filter(similarity_matrix, edge_threshold)
    for edge in edges:
        G.add_edge(edge[0], edge[1], weight=similarity_matrix[edge[0]][edge[1]].item())
        
    print(G.number_of_nodes())
    print(G.number_of_edges())
    
    # if (G.number_of_nodes() > 500000):
    #     exit(1)
    
    # JSON 형식으로 저장
    graph_data = nx.node_link_data(G)
    with open(file_path, 'w') as file:
        json.dump(graph_data, file, cls=CustomJSONEncoder)
    
        
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)
        
def threshold_filter(similarity_matrix, edge_threshold):
    # Ensure the input is converted to a PyTorch tensor
    if not isinstance(similarity_matrix, torch.Tensor):
        similarity_tensor = torch.from_numpy(similarity_matrix).cuda("cuda:1")  # Convert from NumPy array to tensor
    else:
        similarity_tensor = similarity_matrix.cuda("cuda:1")  # Already a tensor, move to GPU

    # Calculate edges based on the threshold
    edges = torch.nonzero(similarity_tensor > edge_threshold, as_tuple=False)
    
    return edges.cpu().numpy()  # Move back to CPU for NetworkX compatibility

    
def load_graph(file_path):
    # Load graph
    # Input: File Path
    # Output: NetworkX Graph
    
    # with open(file_path, 'r') as f:
    #     for i, l in enumerate(f):
    #         print(l)
    #         if i > 10:
    #             break
    
    G = nx.readwrite.json_graph.node_link_graph(json.load(open(file_path)))
    return G

def make_dummy_dataset():
    # Make dummy dataset
    # Input: None
    # Output: Embedding Tensor, Label Tensor
    
    embeddings = torch.randn(1000, 128)
    labels = torch.randint(0, 10, (1000,))
    
    file_path = '../data/small_dummy.pt'
    torch.save((embeddings, labels), file_path)
    
# make_dummy_dataset()

def label_comparison(G):
    # Input: NetworkX Graph
    # Output: None
    
    same_label_count = 0
    diff_label_count = 0

    for edge in G.edges():
        node1_label = G.nodes[edge[0]]['label']
        node2_label = G.nodes[edge[1]]['label']
        
        if node1_label == node2_label:
            same_label_count += 1
        else:
            diff_label_count += 1
    
    print("Same label count: ", same_label_count)
    print("Diff label count: ", diff_label_count)