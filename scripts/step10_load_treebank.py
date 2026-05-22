# step10_load_treebank.py
# Goal: parse the treebank and extract
# (word, POS tag) pairs from every sentence

import conllu

train_path = r"C:\Users\gupta\Downloads\HindiWN_1_5\Hindi_Wordnet_project\data\hi_hdtb-ud-train.conllu"
dev_path   = r"C:\Users\gupta\Downloads\HindiWN_1_5\Hindi_Wordnet_project\data\hi_hdtb-ud-dev.conllu"
test_path  = r"C:\Users\gupta\Downloads\HindiWN_1_5\Hindi_Wordnet_project\data\hi_hdtb-ud-test.conllu"

def load_conllu(path):
    """
    Load a conllu file and return list of sentences.
    Each sentence = list of (word, POS) pairs.
    """
    sentences = []

    with open(path, "r", encoding="utf-8") as f:
        data = f.read()

    parsed = conllu.parse(data)

    for sentence in parsed:
        word_pos_pairs = []
        for token in sentence:
            # skip multi-word tokens
            if isinstance(token["id"], tuple):
                continue
            word = token["form"]   # the actual Hindi word
            pos  = token["upos"]   # the POS tag
            if word and pos:
                word_pos_pairs.append((word, pos))
        if word_pos_pairs:
            sentences.append(word_pos_pairs)

    return sentences

print("Loading treebank files...")
train_data = load_conllu(train_path)
dev_data   = load_conllu(dev_path)
test_data  = load_conllu(test_path)

print(f"Train sentences : {len(train_data)}")
print(f"Dev sentences   : {len(dev_data)}")
print(f"Test sentences  : {len(test_data)}")
print()

# Count total words
train_words = sum(len(s) for s in train_data)
print(f"Total train words: {train_words}")
print()

# Show POS tag distribution
from collections import Counter
all_pos = [pos for sent in train_data for word, pos in sent]
pos_counts = Counter(all_pos)
print("POS tag distribution in training data:")
for pos, count in pos_counts.most_common():
    print(f"  {pos:10} : {count}")
print()

# Show 3 sample sentences
print("Sample sentences:")
for i, sent in enumerate(train_data[:3]):
    print(f"\nSentence {i+1}:")
    for word, pos in sent:
        print(f"  {word:20} → {pos}")
