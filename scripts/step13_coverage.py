# step13_coverage.py
# Goal: check what % of treebank words are covered by WordNet
# This explains why our improvement is small
# and gives us a key analysis for the paper

import pickle
import conllu
from collections import Counter

emb_path = r"C:\Users\gupta\Downloads\HindiWN_1_5\Hindi_Wordnet_project\outputs\hyperbolic_embeddings.pkl"
with open(emb_path, "rb") as f:
    emb_data = pickle.load(f)
node_to_id = emb_data["node_to_id"]

train_path = r"C:\Users\gupta\Downloads\HindiWN_1_5\Hindi_Wordnet_project\data\hi_hdtb-ud-train.conllu"
test_path  = r"C:\Users\gupta\Downloads\HindiWN_1_5\Hindi_Wordnet_project\data\hi_hdtb-ud-test.conllu"

def load_conllu(path):
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    parsed = conllu.parse(data)
    for sentence in parsed:
        pairs = []
        for token in sentence:
            if isinstance(token["id"], tuple):
                continue
            word = token["form"]
            pos  = token["upos"]
            if word and pos:
                pairs.append((word, pos))
        if pairs:
            sentences.append(pairs)
    return sentences

train_data = load_conllu(train_path)
test_data  = load_conllu(test_path)

print("="*55)
print("WORDNET COVERAGE ANALYSIS")
print("="*55)

for split_name, data in [("Train", train_data), ("Test", test_data)]:
    total_tokens   = 0
    covered_tokens = 0
    covered_by_pos = Counter()
    total_by_pos   = Counter()

    for sentence in data:
        for word, pos in sentence:
            total_tokens += 1
            total_by_pos[pos] += 1
            if word in node_to_id:
                covered_tokens += 1
                covered_by_pos[pos] += 1

    coverage = covered_tokens / total_tokens * 100
    print(f"\n{split_name} set:")
    print(f"  Total tokens   : {total_tokens}")
    print(f"  Covered by WN  : {covered_tokens}")
    print(f"  Coverage       : {coverage:.2f}%")
    print(f"\n  Coverage by POS tag:")
    print(f"  {'Tag':<10} {'Covered':>8} {'Total':>8} {'Coverage':>10}")
    print(f"  {'-'*40}")
    for tag in sorted(total_by_pos.keys()):
        cov = covered_by_pos[tag] / total_by_pos[tag] * 100
        print(f"  {tag:<10} {covered_by_pos[tag]:>8} "
              f"{total_by_pos[tag]:>8} {cov:>9.2f}%")
