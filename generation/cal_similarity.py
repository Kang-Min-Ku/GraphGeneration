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
    normalize_similarity_matrix = normalize_similarity(similarity_matrix)
    
    return normalize_similarity_matrix.cpu()

def cal_euclidean_similarity(embeddings, epsilon=1e-9):
    # Calculate similarity (euclidean similarity)
    # Input: Embedding Tensor
    # Output: Similarity Tensor
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = embeddings.to(device)
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    similarity_matrix = torch.cdist(normalized_embeddings, normalized_embeddings, p=2)
    normalize_similarity_matrix = normalize_similarity(similarity_matrix)
    # 모든 값들을 1에서 빼기
    modified_similarity_matrix = 1 - normalize_similarity_matrix
    
    return modified_similarity_matrix.cpu()

def normalize_similarity(similarity_matrix):
    # Normalize similarity matrix
    # Input: Similarity Tensor
    # Output: Normalized Similarity Tensor
    
    min_val = torch.min(similarity_matrix)
    max_val = torch.max(similarity_matrix)
    
    normalized_similarity_matrix = (similarity_matrix - min_val) / (max_val - min_val)
    return normalized_similarity_matrix