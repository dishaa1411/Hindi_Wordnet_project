

# step6_train_embeddings.py (faster version)
# We process edges in batches instead of one by one

import torch
import geoopt
import random
import pickle

print("Loading edges...")

from config import EDGES_CLEAN, EMB_PKL
edges_path = EDGES_CLEAN
save_path  = EMB_PKL

edges = []
nodes = set()

with open(edges_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parent, child = line.split("\t")
        edges.append((parent, child))
        nodes.add(parent)
        nodes.add(child)

print(f"Edges loaded : {len(edges)}")
print(f"Nodes loaded : {len(nodes)}")

# Convert node IDs to numbers
node_list  = sorted(list(nodes))
node_to_id = {node: i for i, node in enumerate(node_list)}
id_to_node = {i: node for node, i in node_to_id.items()}
num_nodes  = len(node_list)

# Convert edges to number pairs
edge_ids = [(node_to_id[p], node_to_id[c]) for p, c in edges]
edge_set = set(edge_ids)

print(f"Total unique nodes: {num_nodes}")
print()

# ---- Model setup ----
DIMS        = 10
EPOCHS      = 50
LEARN_RATE  = 0.01
BATCH_SIZE  = 512    # process 512 edges at a time instead of 1
NEG_SAMPLES = 5

manifold   = geoopt.PoincareBall()
embeddings = geoopt.ManifoldParameter(
    torch.randn(num_nodes, DIMS) * 0.01,
    manifold=manifold
)
optimizer = geoopt.optim.RiemannianAdam([embeddings], lr=LEARN_RATE)

print(f"Starting training...")
print(f"  Epochs     : {EPOCHS}")
print(f"  Batch size : {BATCH_SIZE}")
print(f"  Dimensions : {DIMS}")
print()

# ---- Training loop (batched) ----
edge_tensor = torch.tensor(edge_ids, dtype=torch.long)

for epoch in range(EPOCHS):
    # shuffle edges
    perm       = torch.randperm(len(edge_tensor))
    edge_tensor = edge_tensor[perm]
    total_loss = 0.0
    num_batches = 0

    for i in range(0, len(edge_tensor), BATCH_SIZE):
        batch = edge_tensor[i : i + BATCH_SIZE]

        parents  = batch[:, 0]
        children = batch[:, 1]

        # positive distances (real edges)
        parent_vecs = embeddings[parents]
        child_vecs  = embeddings[children]
        pos_dists   = manifold.dist(parent_vecs, child_vecs)

        # negative distances (fake edges)
        neg_children = torch.randint(0, num_nodes, (len(batch), NEG_SAMPLES))
        neg_vecs     = embeddings[neg_children]
        parent_vecs_expanded = parent_vecs.unsqueeze(1).expand_as(neg_vecs)
        neg_dists    = manifold.dist(parent_vecs_expanded, neg_vecs)

        # loss: real should be closer than fake
        loss = (pos_dists.unsqueeze(1) - neg_dists).clamp(min=0).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss  += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1:3d}/{EPOCHS} | Loss: {avg_loss:.4f}")

print("\nTraining complete!")

# ---- Save ----
output = {
    "embeddings": embeddings.detach().cpu(),
    "node_to_id": node_to_id,
    "id_to_node": id_to_node,
    "dims": DIMS
}


with open(save_path, "wb") as f:
    pickle.dump(output, f)

print(f"Embeddings saved to: {save_path}")
print(f"Each of {num_nodes} synsets now has a {DIMS}-dimensional vector!")




