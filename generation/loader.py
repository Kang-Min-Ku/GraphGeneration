import torch

def load_dataset(file_path):
    # .pt file loader
    embeddings, labels = torch.load(file_path)
    
    # Extract Embedding, Label 
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.int32)
    
    return embeddings_tensor, labels_tensor
