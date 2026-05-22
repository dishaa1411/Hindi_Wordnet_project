# step4_build_graph.py
# Goal: load the edges we saved and build a proper graph
# Then analyze it to make sure it's ready for hyperbolic embedding

import networkx as nx
from collections import Counter

# Load the edges file we created in Phase 1
edges_path  = r"C:\Users\gupta\Downloads\HindiWN_1_5\Hindi_Wordnet_project\outputs\wordnet_edges.tsv"

# Build the graph
G = nx.DiGraph()  # DiGraph = Directed Graph (edges have direction)

with open(edges_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parent, child = line.split("\t")
        G.add_edge(parent, child)

# ---- Print summary ----
print(f"Number of nodes (synsets) : {G.number_of_nodes()}")
print(f"Number of edges (relations): {G.number_of_edges()}")
print()

# Check if it's a proper tree-like structure
print(f"Is the graph a DAG (tree-like)? : {nx.is_directed_acyclic_graph(G)}")
print()

# Find root nodes (nodes with no parents = top of hierarchy)
roots = [n for n in G.nodes() if G.in_degree(n) == 0]
print(f"Number of root nodes (top of hierarchy): {len(roots)}")
print(f"Sample roots: {roots[:5]}")
print()

# Find leaf nodes (nodes with no children = bottom of hierarchy)
leaves = [n for n in G.nodes() if G.out_degree(n) == 0]
print(f"Number of leaf nodes (bottom of hierarchy): {len(leaves)}")
print()

# How deep is the hierarchy?
print("Analyzing depth of hierarchy...")
depths = nx.single_source_shortest_path_length(G, roots[0])
max_depth = max(depths.values())
print(f"Maximum depth from first root: {max_depth}")

