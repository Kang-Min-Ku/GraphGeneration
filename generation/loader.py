import torch
import networkx as nx

def load_dataset(file_path):
    # .pt file loader
    embeddings, labels = torch.load(file_path)
    
    # Extract Embedding, Label 
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).clone().detach()
    labels_tensor = torch.tensor(labels, dtype=torch.int32).clone().detach()
    
    return embeddings_tensor, labels_tensor

def make_dummy_dataset():
    # Make dummy dataset
    # Input: None
    # Output: Embedding Tensor, Label Tensor
    
    embeddings = torch.randn(1000, 128)
    labels = torch.randint(0, 10, (1000,))
    
    file_path = '../data/small_dummy.pt'
    torch.save((embeddings, labels), file_path)
    
# make_dummy_dataset()

def save_graph(similarity_matrix, labels, edge_threshold, file_path):
    # Save graph
    # Input: Similarity Tensor, Label Tensor, Edge Threshold, File Path
    # Output: None
    
    G = nx.Graph()

    for idx, embedding in enumerate(similarity_matrix):
        G.add_node(idx, embedding=embedding, label=labels[idx].item())

    num_nodes = similarity_matrix.shape[0]
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            similarity_value = similarity_matrix[i][j].item()
            
            if similarity_matrix[i][j] > edge_threshold:
                G.add_edge(i, j, weight=similarity_value)
    
    print(G.number_of_edges())
    print(G.number_of_nodes())
    
    nx.write_gml(G, file_path)