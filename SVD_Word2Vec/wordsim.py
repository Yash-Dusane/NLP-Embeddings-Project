import argparse
import os
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Evaluate word embeddings using WordSim-353 dataset.")
parser.add_argument("embedding_path", type=str, help="Path to the word embedding file (.pt)")
args = parser.parse_args()

embedding_filename = os.path.splitext(os.path.basename(args.embedding_path))[0]
output_path = f"{embedding_filename}.csv"

# Load word embeddings
embeddings = torch.load(args.embedding_path, weights_only=False)
word_to_id = embeddings["word_to_id"]  # Dictionary mapping words to indices
word_vectors = embeddings["word_vectors"]  # Word vectors (numpy array or tensor)

# Load WordSim-353 dataset
wordsim_path = "wordsim353crowd.csv"
wordsim_data = pd.read_csv(wordsim_path)
cosine_similarities = []
provided_scores = []
results = []

for _, row in wordsim_data.iterrows():
    word1, word2, score = row["Word 1"].lower(), row["Word 2"].lower(), row["Human (Mean)"]
    if word1 in word_to_id and word2 in word_to_id:
        vec1 = word_vectors[word_to_id[word1]].reshape(1, -1)
        vec2 = word_vectors[word_to_id[word2]].reshape(1, -1)
        cosine_similarity_score = cosine_similarity(vec1, vec2)[0][0]
        cosine_similarities.append(cosine_similarity_score)
        provided_scores.append(score)
        results.append([word1, word2, cosine_similarity_score])

# Compute Spearman's Rank Correlation
spearman_corr, _ = spearmanr(cosine_similarities, provided_scores)
print(f"Spearman's Rank Correlation: {spearman_corr}")

# Save results to CSV
results_df = pd.DataFrame(results, columns=["word1", "word2", "cosine_similarity"])
results_df.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")


# # Scatter plot for Cosine Similarity vs Human Mean
# plt.figure(figsize=(8, 6))
# plt.scatter(provided_scores, cosine_similarities, color='blue', edgecolors='black', s=100)

# # Labeling the plot
# plt.title("SkipGram", fontsize=14)
# plt.xlabel("Human Mean (WordSim-353)", fontsize=12)
# plt.ylabel("Cosine Similarity (Embedding)", fontsize=12)
# plt.grid(True)

# # Save the plot as an image file
# plt.tight_layout()
# # plt.savefig(args.plot_path)
# plt.show()


# # Histogram of Cosine Similarities
# plt.figure(figsize=(8, 6))
# plt.hist(cosine_similarities, bins=20, color='blue', edgecolor='black', alpha=0.7)

# # Labeling the plot
# plt.title("Histogram - SVD", fontsize=14)
# plt.xlabel("Cosine Similarity", fontsize=12)
# plt.ylabel("Frequency", fontsize=12)
# plt.grid(True)

# # Save the histogram as an image file
# plt.tight_layout()
# # plt.savefig(args.histogram_path)
# plt.show()

