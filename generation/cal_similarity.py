import torch
import torch.nn.functional as F

def cal_cosine_similarity(embeddings):
    # Calculate similarity (cosine similarity)
    # Input: Embedding Tensor
    # Output: Similarity Tensor
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = embeddings.to(device)
    
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    similarity_matrix = torch.matmul(normalized_embeddings, normalized_embeddings.transpose(0, 1))
    return similarity_matrix.cpu()

def cal_euclidean_similarity(embeddings):
    # Calculate similarity (euclidean similarity)
    # Input: Embedding Tensor
    # Output: Similarity Tensor
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = embeddings.to(device)
    
    squared_norm = torch.sum(embeddings**2, dim=1, keepdim=True)
    similarity_matrix = torch.mm(embeddings, embeddings.t()) / torch.sqrt(squared_norm.mm(squared_norm.t()))
    
    return similarity_matrix.cpu()