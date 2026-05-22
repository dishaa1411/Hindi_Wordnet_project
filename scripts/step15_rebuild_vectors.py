# step15_rebuild_vectors.py
# Goal: rebuild word vectors directly from WordNet + embeddings
# Skips the heavy pkl file entirely

import torch
import pickle

print("Loading embeddings...")
emb_path = r"C:\Users\gupta\Downloads\HindiWN_1_5\Hindi_Wordnet_project\outputs\hyperbolic_embeddings.pkl"
with open(emb_path, "rb") as f:
    emb_data = pickle.load(f)

embeddings_matrix = emb_data["embeddings"]
node_to_id        = emb_data["node_to_id"]
print(f"Embeddings loaded: {embeddings_matrix.shape}")

# Parse WordNet and build word → vector directly as lists
print("Building word vectors...")
data_path = r"C:\Users\gupta\Downloads\HindiWN_1_5\Hindi_Wordnet_project\database\data_txt"

word_to_vector = {}

with open(data_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or "|" not in line:
            continue
        data_part, _ = line.split("|", 1)
        tokens = data_part.strip().split()
        if len(tokens) < 4:
            continue

        synset_id = tokens[0]
        words     = tokens[3].split(":")

        if synset_id in node_to_id:
            idx    = node_to_id[synset_id]
            # store as plain list immediately — no tensor objects
            vector = embeddings_matrix[idx].tolist()
            for word in words:
                word = word.strip()
                if word:
                    word_to_vector[word] = vector

print(f"Words mapped: {len(word_to_vector)}")

# Save as plain dict of lists — very light
save_path = r"C:\Users\gupta\Downloads\HindiWN_1_5\Hindi_Wordnet_project\outputs\word_vectors_light.pkl"
with open(save_path, "wb") as f:
    pickle.dump(word_to_vector, f)

print(f"Saved to word_vectors_light.pkl")

# Quick memory check
import sys
size_mb = sys.getsizeof(word_to_vector) / 1024 / 1024
print(f"Dict size in memory: ~{size_mb:.1f} MB")
