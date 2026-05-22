# step14_fix_coverage.py
# Goal: build a proper word → hyperbolic vector lookup
# The problem: WordNet nodes are synset IDs, not words
# The fix: parse WordNet again and map each WORD to its synset's embedding

import pickle
import torch

# Load existing embeddings (synset_id → vector)
emb_path = r"C:\Users\gupta\Downloads\HindiWN_1_5\Hindi_Wordnet_project\outputs\hyperbolic_embeddings.pkl"
with open(emb_path, "rb") as f:
    emb_data = pickle.load(f)

embeddings_matrix = emb_data["embeddings"]
node_to_id        = emb_data["node_to_id"]  # synset_id → index
id_to_node        = emb_data["id_to_node"]  # index → synset_id

print(f"Loaded {len(node_to_id)} synset embeddings")

# Parse WordNet data file to get word → synset_id mapping
data_path = r"C:\Users\gupta\Downloads\HindiWN_1_5\Hindi_Wordnet_project\database\data_txt"

pos_map = {
    "01": "noun",
    "02": "adjective",
    "03": "verb",
    "04": "adverb"
}

word_to_vector = {}  # word → hyperbolic vector
word_to_pos    = {}  # word → POS label
words_found    = 0
words_missed   = 0

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
        pos_code  = tokens[1]
        pos_label = pos_map.get(pos_code, "unknown")
        words     = tokens[3].split(":")

        # Look up this synset's embedding
        if synset_id in node_to_id:
            idx    = node_to_id[synset_id]
            vector = embeddings_matrix[idx]

            # Map every word in this synset to the vector
            for word in words:
                word = word.strip()
                if word:
                    word_to_vector[word] = vector
                    word_to_pos[word]    = pos_label
                    words_found += 1
        else:
            words_missed += len(words)

print(f"Words mapped to vectors : {len(word_to_vector)}")
print(f"Words found in graph    : {words_found}")
print(f"Words missed            : {words_missed}")
print()

# Save the word → vector mapping
output = {
    "word_to_vector": word_to_vector,
    "word_to_pos":    word_to_pos,
    "dims":           emb_data["dims"]
}

save_path = r"C:\Users\gupta\Downloads\HindiWN_1_5\Hindi_Wordnet_project\outputs\word_vectors.pkl"
with open(save_path, "wb") as f:
    pickle.dump(output, f)

print(f"Word vectors saved to: {save_path}")
print()

# Now check coverage on treebank
import conllu
from collections import Counter

test_path = r"C:\Users\gupta\Downloads\HindiWN_1_5\Hindi_Wordnet_project\data\hi_hdtb-ud-test.conllu"

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

test_data = load_conllu(test_path)

print("="*55)
print("COVERAGE AFTER FIX")
print("="*55)

total    = 0
covered  = 0
by_pos_covered = Counter()
by_pos_total   = Counter()

for sentence in test_data:
    for word, pos in sentence:
        total += 1
        by_pos_total[pos] += 1
        if word in word_to_vector:
            covered += 1
            by_pos_covered[pos] += 1

print(f"\nTotal tokens  : {total}")
print(f"Covered       : {covered}")
print(f"Coverage      : {covered/total*100:.2f}%")
print()
print(f"{'Tag':<10} {'Covered':>8} {'Total':>8} {'Coverage':>10}")
print("-"*40)
for tag in sorted(by_pos_total.keys()):
    cov = by_pos_covered[tag] / by_pos_total[tag] * 100
    print(f"{tag:<10} {by_pos_covered[tag]:>8} "
          f"{by_pos_total[tag]:>8} {cov:>9.2f}%")
