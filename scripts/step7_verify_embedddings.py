# step7_verify_embeddings.py
# Goal: verify our embeddings make sense
# We'll check: can the model tell that related words are closer
# than unrelated words in hyperbolic space?

import torch
import geoopt
import pickle

# Load the saved embeddings
save_path = r"C:\Users\gupta\Downloads\HindiWN_1_5\Hindi_Wordnet_project\outputs\hyperbolic_embeddings.pkl"

with open(save_path, "rb") as f:
    data = pickle.load(f)

embeddings = data["embeddings"]
node_to_id = data["node_to_id"]
id_to_node = data["id_to_node"]

manifold = geoopt.PoincareBall()

print(f"Embeddings shape: {embeddings.shape}")
print(f"  → {embeddings.shape[0]} synsets, each with {embeddings.shape[1]} dimensions")
print()

# ---- Check 1: Norm of each vector ----
# In Poincaré disk, norm tells us depth in hierarchy
# norm close to 0 = near center = general word
# norm close to 1 = near edge   = specific word
norms = embeddings.norm(dim=1)
print(f"Vector norms (0=general, 1=specific):")
print(f"  Minimum norm : {norms.min():.4f}")
print(f"  Maximum norm : {norms.max():.4f}")
print(f"  Average norm : {norms.mean():.4f}")
print()

# ---- Check 2: Find most general synsets ----
# (closest to center = smallest norm)
print("Top 5 most GENERAL synsets (near center of disk):")
general_ids = norms.argsort()[:5]
for idx in general_ids:
    node_id = id_to_node[idx.item()]
    print(f"  Synset {node_id} | norm: {norms[idx]:.4f}")
print()

# ---- Check 3: Find most specific synsets ----
# (closest to edge = largest norm)
print("Top 5 most SPECIFIC synsets (near edge of disk):")
specific_ids = norms.argsort(descending=True)[:5]
for idx in specific_ids:
    node_id = id_to_node[idx.item()]
    print(f"  Synset {node_id} | norm: {norms[idx]:.4f}")
print()

# ---- Check 4: Test a real edge vs fake edge ----
# Load edges to test with
edges_path = r"C:\Users\gupta\Downloads\HindiWN_1_5\Hindi_Wordnet_project\outputs\wordnet_edges_clean.tsv"

edges = []
with open(edges_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parent, child = line.split("\t")
        edges.append((parent, child))

print("Testing real edges vs fake edges:")
print("(real edges should have SMALLER distance)")
print()

import random
for i in range(5):
    # real edge
    real_parent, real_child = edges[i]
    p_id = node_to_id[real_parent]
    c_id = node_to_id[real_child]
    real_dist = manifold.dist(embeddings[p_id], embeddings[c_id]).item()

    # fake edge (random pair)
    fake_child = random.choice(list(node_to_id.keys()))
    f_id = node_to_id[fake_child]
    fake_dist = manifold.dist(embeddings[p_id], embeddings[f_id]).item()

    status = "✅ GOOD" if real_dist < fake_dist else "❌ BAD"
    print(f"  {status} | Real dist: {real_dist:.4f} | Fake dist: {fake_dist:.4f}")
