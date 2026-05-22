# step3_relations.py
# Goal: extract hypernym/hyponym relations between synsets
# These relations = edges in our graph for Poincaré embedding

from config import DATA_TXT, EDGES_TSV
file_path   = DATA_TXT

# Relation codes from Hindi WordNet
relation_map = {
    "1102": "hypernym",    # parent (more general)
    "1103": "hyponym",     # child (more specific)
    "1223": "hypernym",
    "1224": "hyponym",
    "2111": "hypernym",
    "2224": "hyponym",
    "1141": "holonym",     # part of
    "1142": "meronym",     # has part
}

pos_map = {
    "01": "noun",
    "02": "adjective",
    "03": "verb",
    "04": "adverb"
}

synsets = []   # stores synset info
edges = []     # stores (parent, child) pairs for our graph

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or "|" not in line:
            continue

        data_part, definition = line.split("|", 1)
        tokens = data_part.strip().split()

        synset_id = tokens[0]
        pos_code  = tokens[1]
        pos_label = pos_map.get(pos_code, "unknown")
        words     = tokens[3].split(":")

        # store synset
        synsets.append({
            "id": synset_id,
            "pos": pos_label,
            "words": words,
        })

        # now extract relations
        # relations start at token index 5 onwards
        # they come in pairs: (relation_code, related_synset_id)
        i = 5
        while i < len(tokens) - 1:
            rel_code     = tokens[i]
            related_id   = tokens[i + 1]
            rel_name     = relation_map.get(rel_code, None)

            if rel_name == "hypernym" and related_id != "0000":
                # hypernym means: related_id is the PARENT of synset_id
                edges.append((related_id, synset_id))

            i += 2

# ---- Print summary ----
print(f"Total synsets : {len(synsets)}")
print(f"Total edges   : {len(edges)}")
print()

# POS breakdown
from collections import Counter
pos_counts = Counter(s["pos"] for s in synsets)
print("POS breakdown:")
for pos, count in pos_counts.items():
    print(f"  {pos}: {count}")
print()

# Sample edges
print("Sample edges (parent → child):")
for parent, child in edges[:5]:
    print(f"  {parent} → {child}")

# Save edges to file for Phase 2
output_path = EDGES_TSV

with open(output_path, "w", encoding="utf-8") as f:
    for parent, child in edges:
        f.write(f"{parent}\t{child}\n")

print(f"\nEdges saved to: {output_path}")
