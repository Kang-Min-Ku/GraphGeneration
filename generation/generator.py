from loader import load_dataset
from cal_similarity import cal_cosine_similarity, cal_euclidean_similarity
import torch

# Load Dataset
embeddings, labels = load_dataset('embeddings.pt')

# Calculate Similarity
cosine_similarity = cal_cosine_similarity(embeddings)
euclidean_similarity = cal_euclidean_similarity(embeddings)

# Save Similarity
