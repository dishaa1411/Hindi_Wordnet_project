# step5_fix_graph.py
# Goal: fix the graph issues we found:
#   1. Remove cycles (makes it a proper DAG) directed acyclic graph
#   2. Understand the 296 separate trees
#   3. Check real depth properly

import networkx as nx
from collections import Counter

from config import EDGES_TSV, EDGES_CLEAN
edges_path  = EDGES_TSV
output_path = EDGES_CLEAN

# Load edges
G = nx.DiGraph()
with open(edges_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parent, child = line.split("\t")
        G.add_edge(parent, child)

print(f"Before fixing:")
print(f"  Nodes : {G.number_of_nodes()}")
print(f"  Edges : {G.number_of_edges()}")
print(f"  Is DAG: {nx.is_directed_acyclic_graph(G)}")
print()

# ---- Fix 1: Remove cycles ----
# Find all cycles and remove one edge from each
print("Fixing cycles...")
cycles_removed = 0
while not nx.is_directed_acyclic_graph(G):
    try:
        # find one cycle
        cycle = nx.find_cycle(G, orientation="original")
        # remove the last edge in that cycle
        edge_to_remove = cycle[-1][:2]
        G.remove_edge(*edge_to_remove)
        cycles_removed += 1
    except nx.NetworkXNoCycle:
        break

print(f"  Cycles removed: {cycles_removed}")
print(f"  Is DAG now: {nx.is_directed_acyclic_graph(G)}")
print()

# ---- Fix 2: Analyze connected components ----
# Convert to undirected to find connected groups
undirected = G.to_undirected()
components = list(nx.connected_components(undirected))
print(f"Number of separate trees: {len(components)}")

# Sort by size to see the biggest ones
component_sizes = sorted([len(c) for c in components], reverse=True)
print(f"Sizes of top 5 trees: {component_sizes[:5]}")
print(f"Sizes of smallest 5 trees: {component_sizes[-5:]}")
print()

# ---- Fix 3: Keep only the largest components ----
# Small isolated components (size < 3) are usually noise
big_components = [c for c in components if len(c) >= 3]
nodes_to_keep  = set()
for c in big_components:
    nodes_to_keep.update(c)

# Build cleaned graph with only good nodes
G_clean = G.subgraph(nodes_to_keep).copy()
print(f"After keeping components with 3+ nodes:")
print(f"  Nodes : {G_clean.number_of_nodes()}")
print(f"  Edges : {G_clean.number_of_edges()}")
print()

# ---- Fix 4: Check real depth properly ----
roots = [n for n in G_clean.nodes() if G_clean.in_degree(n) == 0]
print(f"Root nodes: {len(roots)}")

# Check depth from ALL roots, not just one
all_depths = []
for root in roots:
    depths = nx.single_source_shortest_path_length(G_clean, root)
    all_depths.extend(depths.values())

print(f"Maximum depth across all trees : {max(all_depths)}")
print(f"Average depth                  : {sum(all_depths)/len(all_depths):.2f}")
print()

# ---- Save the cleaned graph ----

with open(output_path, "w", encoding="utf-8") as f:
    for parent, child in G_clean.edges():
        f.write(f"{parent}\t{child}\n")

print(f"Clean graph saved to: {output_path}")
