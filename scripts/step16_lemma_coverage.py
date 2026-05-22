# step16_lemma_coverage.py
# Goal: use lemmas instead of word forms for WordNet lookup
# The treebank already has lemmas in column 3 — we just need to use them!

import conllu
import pickle
from collections import Counter
from config import WORD_VEC_PKL, TRAIN_CONLLU, DEV_CONLLU, TEST_CONLLU

# Load word vectors
with open(WORD_VEC_PKL, "rb") as f:
    word_to_vector = pickle.load(f)
print(f"Word vectors loaded: {len(word_to_vector)}")

# Load treebank WITH lemmas this time
def load_conllu_with_lemmas(path):
    """
    Returns list of sentences.
    Each sentence = list of (word, lemma, POS) triples.
    """
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    parsed = conllu.parse(data)
    for sentence in parsed:
        triples = []
        for token in sentence:
            if isinstance(token["id"], tuple):
                continue
            word  = token["form"]    # inflected form e.g. मस्जिदों
            lemma = token["lemma"]   # base form    e.g. मस्जिद
            pos   = token["upos"]
            if word and pos:
                triples.append((word, lemma, pos))
        if triples:
            sentences.append(triples)
    return sentences

print("Loading treebank with lemmas...")
train_data = load_conllu_with_lemmas(TRAIN_CONLLU)
test_data  = load_conllu_with_lemmas(TEST_CONLLU)

# Check coverage using WORD form (old way)
def check_coverage(data, use_lemma=False):
    total   = 0
    covered = 0
    by_pos_covered = Counter()
    by_pos_total   = Counter()

    for sentence in data:
        for word, lemma, pos in sentence:
            total += 1
            by_pos_total[pos] += 1
            lookup = lemma if use_lemma else word
            if lookup in word_to_vector:
                covered += 1
                by_pos_covered[pos] += 1

    overall = covered / total * 100
    return overall, by_pos_covered, by_pos_total

print()
print("="*55)
print("COVERAGE COMPARISON")
print("="*55)

# Old way: using word form
word_cov, word_by_pos, total_by_pos = check_coverage(
    test_data, use_lemma=False)

# New way: using lemma
lemma_cov, lemma_by_pos, _ = check_coverage(
    test_data, use_lemma=True)

print(f"\n{'Method':<25} {'Coverage':>10}")
print("-"*40)
print(f"{'Word form (old way)':<25} {word_cov:>9.2f}%")
print(f"{'Lemma (new way)':<25} {lemma_cov:>9.2f}%")
print(f"{'Improvement':<25} {lemma_cov-word_cov:>+9.2f}%")

print(f"\n\nPer-tag coverage comparison:")
print(f"{'Tag':<10} {'Word form':>10} {'Lemma':>10} {'Diff':>10}")
print("-"*45)
for tag in sorted(total_by_pos.keys()):
    w_cov = word_by_pos[tag]  / total_by_pos[tag] * 100
    l_cov = lemma_by_pos[tag] / total_by_pos[tag] * 100
    diff  = l_cov - w_cov
    print(f"{tag:<10} {w_cov:>9.2f}% {l_cov:>9.2f}% {diff:>+9.2f}%")

# Save lemma lookup info for retraining
print("\nSaving lemma mapping from treebank...")
word_to_lemma = {}
for sentence in train_data:
    for word, lemma, pos in sentence:
        word_to_lemma[word] = lemma

save_path = r"C:\Users\gupta\Downloads\HindiWN_1_5\Hindi_Wordnet_project\outputs\word_to_lemma.pkl"
with open(save_path, "wb") as f:
    pickle.dump(word_to_lemma, f)

print(f"Word to lemma mapping saved: {len(word_to_lemma)} words")
