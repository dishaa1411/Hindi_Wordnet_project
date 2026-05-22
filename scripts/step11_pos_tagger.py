# step11_pos_tagger.py
# Goal: train a BiLSTM-CRF POS tagger using our hyperbolic embeddings

import torch
import torch.nn as nn
import pickle
import conllu
import geoopt
from torchcrf import CRF
from collections import Counter
from torch.utils.data import Dataset, DataLoader

# ---- Step 1: Load hyperbolic embeddings ----
print("Loading hyperbolic embeddings...")
emb_path = r"C:\Users\gupta\Downloads\HindiWN_1_5\Hindi_Wordnet_project\outputs\hyperbolic_embeddings.pkl"

with open(emb_path, "rb") as f:
    emb_data = pickle.load(f)

embeddings_matrix = emb_data["embeddings"]  # shape: [33662, 10]
node_to_id        = emb_data["node_to_id"]
print(f"Embeddings loaded: {embeddings_matrix.shape}")

# ---- Step 2: Load treebank ----
print("Loading treebank...")
train_path = r"C:\Users\gupta\Downloads\HindiWN_1_5\Hindi_Wordnet_project\data\hi_hdtb-ud-train.conllu"
dev_path   = r"C:\Users\gupta\Downloads\HindiWN_1_5\Hindi_Wordnet_project\data\hi_hdtb-ud-dev.conllu"

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
dev_data   = load_conllu(dev_path)
print(f"Train: {len(train_data)} sentences")
print(f"Dev  : {len(dev_data)} sentences")

# ---- Step 3: Build vocabularies ----
# Word vocabulary
all_words = [word for sent in train_data for word, pos in sent]
word_counts = Counter(all_words)
word_to_id = {"<PAD>": 0, "<UNK>": 1}
for word, count in word_counts.items():
    if count >= 2:  # only keep words seen at least twice
        word_to_id[word] = len(word_to_id)

# POS tag vocabulary
all_tags = [pos for sent in train_data for word, pos in sent]
tag_list = sorted(set(all_tags))
tag_to_id = {tag: i for i, tag in enumerate(tag_list)}
id_to_tag = {i: tag for tag, i in tag_to_id.items()}

print(f"Vocabulary size : {len(word_to_id)}")
print(f"Number of tags  : {len(tag_to_id)}")
print(f"Tags            : {list(tag_to_id.keys())}")

# ---- Step 4: Dataset class ----
class POSDataset(Dataset):
    def __init__(self, sentences, word_to_id, tag_to_id):
        self.sentences = sentences
        self.word_to_id = word_to_id
        self.tag_to_id  = tag_to_id

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        word_ids = [self.word_to_id.get(w, 1) for w, p in sentence]
        tag_ids  = [self.tag_to_id[p] for w, p in sentence]
        return torch.tensor(word_ids), torch.tensor(tag_ids)

def collate_fn(batch):
    """Pad sentences to same length in each batch."""
    word_seqs, tag_seqs = zip(*batch)
    lengths   = [len(s) for s in word_seqs]
    max_len   = max(lengths)
    word_pad  = torch.zeros(len(batch), max_len, dtype=torch.long)
    tag_pad   = torch.zeros(len(batch), max_len, dtype=torch.long)
    mask      = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, (w, t) in enumerate(zip(word_seqs, tag_seqs)):
        word_pad[i, :len(w)] = w
        tag_pad[i,  :len(t)] = t
        mask[i,     :len(w)] = True
    return word_pad, tag_pad, mask

train_dataset = POSDataset(train_data, word_to_id, tag_to_id)
dev_dataset   = POSDataset(dev_data,   word_to_id, tag_to_id)
train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True,  collate_fn=collate_fn)
dev_loader    = DataLoader(dev_dataset,   batch_size=32, shuffle=False, collate_fn=collate_fn)

# ---- Step 5: BiLSTM-CRF Model ----
class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, num_tags, emb_dim, hidden_dim, hyp_dim):
        super().__init__()

        # Word embedding layer (learned from scratch)
        self.word_embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        # BiLSTM: takes word embedding + hyperbolic embedding as input
        self.lstm = nn.LSTM(
            input_size  = emb_dim + hyp_dim,  # word emb + hyperbolic emb
            hidden_size = hidden_dim,
            num_layers  = 2,
            batch_first = True,
            bidirectional = True,
            dropout     = 0.3
        )

        # Linear layer: maps BiLSTM output to tag scores
        self.linear = nn.Linear(hidden_dim * 2, num_tags)

        # CRF layer: makes final tag decisions
        self.crf = CRF(num_tags, batch_first=True)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

    def forward(self, word_ids, hyp_embs, mask):
        # Get word embeddings
        word_embs = self.word_embedding(word_ids)
        word_embs = self.dropout(word_embs)

        # Concatenate word embeddings + hyperbolic embeddings
        combined = torch.cat([word_embs, hyp_embs], dim=-1)

        # Pass through BiLSTM
        lstm_out, _ = self.lstm(combined)
        lstm_out    = self.dropout(lstm_out)

        # Get tag scores
        emissions = self.linear(lstm_out)
        return emissions

    def loss(self, emissions, tags, mask):
        return -self.crf(emissions, tags, mask=mask, reduction="mean")

    def predict(self, emissions, mask):
        return self.crf.decode(emissions, mask=mask)

# ---- Step 6: Hyperbolic embedding lookup ----
# For each word in the treebank, look up its hyperbolic embedding
# If the word isn't in WordNet, use a zero vector
HYP_DIM = 10

def get_hyp_embeddings(word_ids_batch, word_to_id, node_to_id, embeddings_matrix):
    """Look up hyperbolic embeddings for a batch of word id sequences."""
    batch_size, seq_len = word_ids_batch.shape
    hyp_embs = torch.zeros(batch_size, seq_len, HYP_DIM)

    id_to_word = {v: k for k, v in word_to_id.items()}

    for i in range(batch_size):
        for j in range(seq_len):
            word_id = word_ids_batch[i, j].item()
            word    = id_to_word.get(word_id, "<UNK>")
            # try to find word in WordNet node list
            node_id = node_to_id.get(word, None)
            if node_id is not None:
                hyp_embs[i, j] = embeddings_matrix[node_id]

    return hyp_embs

# ---- Step 7: Training ----
VOCAB_SIZE = len(word_to_id)
NUM_TAGS   = len(tag_to_id)
EMB_DIM    = 64
HIDDEN_DIM = 128
EPOCHS     = 15
LR         = 0.001

model     = BiLSTMCRF(VOCAB_SIZE, NUM_TAGS, EMB_DIM, HIDDEN_DIM, HYP_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(f"\nStarting training...")
print(f"  Epochs     : {EPOCHS}")
print(f"  Batch size : 32")
print(f"  Hidden dim : {HIDDEN_DIM}")
print()

def evaluate(model, loader):
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for word_ids, tag_ids, mask in loader:
            hyp_embs  = get_hyp_embeddings(word_ids, word_to_id,
                                           node_to_id, embeddings_matrix)
            emissions = model(word_ids, hyp_embs, mask)
            preds     = model.predict(emissions, mask)
            for i, pred_seq in enumerate(preds):
                true_seq = tag_ids[i, :len(pred_seq)].tolist()
                correct += sum(p == t for p, t in zip(pred_seq, true_seq))
                total   += len(pred_seq)
    return correct / total if total > 0 else 0

best_dev_acc = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss  = 0
    num_batches = 0

    for word_ids, tag_ids, mask in train_loader:
        hyp_embs  = get_hyp_embeddings(word_ids, word_to_id,
                                       node_to_id, embeddings_matrix)
        emissions = model(word_ids, hyp_embs, mask)
        loss      = model.loss(emissions, tag_ids, mask)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss  += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    dev_acc  = evaluate(model, dev_loader)

    if dev_acc > best_dev_acc:
        best_dev_acc = dev_acc
        # save best model
        torch.save(model.state_dict(),
            r"C:\Users\gupta\Downloads\HindiWN_1_5\Hindi_Wordnet_project\outputs\best_model.pt")

    print(f"Epoch {epoch+1:2d}/{EPOCHS} | Loss: {avg_loss:.4f} | Dev Acc: {dev_acc*100:.2f}%")

print(f"\nBest Dev Accuracy: {best_dev_acc*100:.2f}%")
print("Model saved!")
