# step17_retrain_lemma.py
# Goal: retrain POS tagger using lemmas for WordNet lookup
# Better coverage = better hyperbolic signal = hopefully better results!

import torch
import torch.nn as nn
import pickle
import conllu
from torchcrf import CRF
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from config import WORD_VEC_PKL, TRAIN_CONLLU, DEV_CONLLU, TEST_CONLLU, OUTPUTS_DIR
import os

print("Loading data...")

# Load word vectors
with open(WORD_VEC_PKL, "rb") as f:
    word_to_vector = pickle.load(f)
print(f"Word vectors loaded: {len(word_to_vector)}")

# Load lemma mapping
lemma_path = os.path.join(OUTPUTS_DIR, "word_to_lemma.pkl")
with open(lemma_path, "rb") as f:
    word_to_lemma = pickle.load(f)
print(f"Lemma mapping loaded: {len(word_to_lemma)} words")

# Load treebank with lemmas
def load_conllu_with_lemmas(path):
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    parsed = conllu.parse(data)
    for sentence in parsed:
        triples = []
        for token in sentence:
            if isinstance(token["id"], tuple):
                continue
            word  = token["form"]
            lemma = token["lemma"]
            pos   = token["upos"]
            if word and pos:
                triples.append((word, lemma, pos))
        if triples:
            sentences.append(triples)
    return sentences

train_data = load_conllu_with_lemmas(TRAIN_CONLLU)
dev_data   = load_conllu_with_lemmas(DEV_CONLLU)
test_data  = load_conllu_with_lemmas(TEST_CONLLU)
print(f"Train: {len(train_data)} | Dev: {len(dev_data)} | Test: {len(test_data)}")

# Vocabularies
all_words   = [w for sent in train_data for w, l, p in sent]
word_counts = Counter(all_words)
word_to_id  = {"<PAD>": 0, "<UNK>": 1}
for word, count in word_counts.items():
    if count >= 2:
        word_to_id[word] = len(word_to_id)

all_tags  = [p for sent in train_data for w, l, p in sent]
tag_list  = sorted(set(all_tags))
tag_to_id = {tag: i for i, tag in enumerate(tag_list)}
id_to_tag = {i: tag for tag, i in tag_to_id.items()}
id_to_word = {v: k for k, v in word_to_id.items()}

HYP_DIM    = 10
VOCAB_SIZE = len(word_to_id)
NUM_TAGS   = len(tag_to_id)
EMB_DIM    = 64
HIDDEN_DIM = 128
print(f"Vocab: {VOCAB_SIZE} | Tags: {NUM_TAGS}")

# Dataset — now stores lemmas too
class POSDataset(Dataset):
    def __init__(self, sentences, word_to_id, tag_to_id):
        self.sentences  = sentences
        self.word_to_id = word_to_id
        self.tag_to_id  = tag_to_id

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        word_ids = [self.word_to_id.get(w, 1) for w, l, p in sentence]
        tag_ids  = [self.tag_to_id[p] for w, l, p in sentence]
        lemmas   = [l for w, l, p in sentence]
        return torch.tensor(word_ids), torch.tensor(tag_ids), lemmas

def collate_fn(batch):
    word_seqs, tag_seqs, lemma_seqs = zip(*batch)
    max_len  = max(len(s) for s in word_seqs)
    word_pad = torch.zeros(len(batch), max_len, dtype=torch.long)
    tag_pad  = torch.zeros(len(batch), max_len, dtype=torch.long)
    mask     = torch.zeros(len(batch), max_len, dtype=torch.bool)
    padded_lemmas = []
    for i, (w, t, l) in enumerate(zip(word_seqs, tag_seqs, lemma_seqs)):
        word_pad[i, :len(w)] = w
        tag_pad[i,  :len(t)] = t
        mask[i,     :len(w)] = True
        # pad lemmas with empty string
        padded_lemmas.append(l + [""] * (max_len - len(l)))
    return word_pad, tag_pad, mask, padded_lemmas

train_dataset = POSDataset(train_data, word_to_id, tag_to_id)
dev_dataset   = POSDataset(dev_data,   word_to_id, tag_to_id)
test_dataset  = POSDataset(test_data,  word_to_id, tag_to_id)
train_loader  = DataLoader(train_dataset, batch_size=32,
                           shuffle=True,  collate_fn=collate_fn)
dev_loader    = DataLoader(dev_dataset,   batch_size=32,
                           shuffle=False, collate_fn=collate_fn)
test_loader   = DataLoader(test_dataset,  batch_size=32,
                           shuffle=False, collate_fn=collate_fn)

# Hyperbolic lookup using LEMMA
def get_hyp_embeddings(word_ids_batch, lemma_batch):
    batch_size, seq_len = word_ids_batch.shape
    hyp_embs = torch.zeros(batch_size, seq_len, HYP_DIM)
    for i in range(batch_size):
        for j in range(seq_len):
            lemma = lemma_batch[i][j]
            # try lemma first, then word form
            if lemma in word_to_vector:
                hyp_embs[i, j] = torch.tensor(
                    word_to_vector[lemma], dtype=torch.float32)
            else:
                word = id_to_word.get(
                    word_ids_batch[i, j].item(), "<UNK>")
                if word in word_to_vector:
                    hyp_embs[i, j] = torch.tensor(
                        word_to_vector[word], dtype=torch.float32)
    return hyp_embs

# Model
class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, num_tags, emb_dim, hidden_dim, hyp_dim):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size    = emb_dim + hyp_dim,
            hidden_size   = hidden_dim,
            num_layers    = 2,
            batch_first   = True,
            bidirectional = True,
            dropout       = 0.3
        )
        self.linear  = nn.Linear(hidden_dim * 2, num_tags)
        self.crf     = CRF(num_tags, batch_first=True)
        self.dropout = nn.Dropout(0.3)

    def forward(self, word_ids, hyp_embs, mask):
        word_embs   = self.dropout(self.word_embedding(word_ids))
        combined    = torch.cat([word_embs, hyp_embs], dim=-1)
        lstm_out, _ = self.lstm(combined)
        emissions   = self.linear(self.dropout(lstm_out))
        return emissions

    def loss(self, emissions, tags, mask):
        return -self.crf(emissions, tags, mask=mask, reduction="mean")

    def predict(self, emissions, mask):
        return self.crf.decode(emissions, mask=mask)

model     = BiLSTMCRF(VOCAB_SIZE, NUM_TAGS, EMB_DIM, HIDDEN_DIM, HYP_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def evaluate(loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for word_ids, tag_ids, mask, lemmas in loader:
            hyp_embs  = get_hyp_embeddings(word_ids, lemmas)
            emissions = model(word_ids, hyp_embs, mask)
            preds     = model.predict(emissions, mask)
            for i, pred_seq in enumerate(preds):
                true_seq = tag_ids[i, :len(pred_seq)].tolist()
                correct += sum(p == t for p, t in zip(pred_seq, true_seq))
                total   += len(pred_seq)
    return correct / total if total > 0 else 0

print(f"\nTraining with lemma-based coverage (53%)...")
print(f"{'Epoch':<8} {'Loss':>8} {'Dev Acc':>10}")
print("-"*30)

best_acc  = 0
save_path = os.path.join(OUTPUTS_DIR, "best_model_v3.pt")

for epoch in range(15):
    model.train()
    total_loss = num_batches = 0
    for word_ids, tag_ids, mask, lemmas in train_loader:
        hyp_embs  = get_hyp_embeddings(word_ids, lemmas)
        emissions = model(word_ids, hyp_embs, mask)
        loss      = model.loss(emissions, tag_ids, mask)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss  += loss.item()
        num_batches += 1

    dev_acc = evaluate(dev_loader)
    if dev_acc > best_acc:
        best_acc = dev_acc
        torch.save(model.state_dict(), save_path)

    print(f"{epoch+1:<8} {total_loss/num_batches:>8.4f} {dev_acc*100:>9.2f}%")

# Final test evaluation
test_acc = evaluate(test_loader)

print(f"\n{'='*55}")
print(f"FINAL COMPARISON — ALL MODELS")
print(f"{'='*55}")
print(f"{'Model':<40} {'Test Acc':>10}")
print(f"{'-'*55}")
print(f"{'Baseline (no hyperbolic)':<40} {'95.52%':>10}")
print(f"{'Ours v1 (0% WN coverage)':<40} {'95.57%':>10}")
print(f"{'Ours v2 (46% coverage, word form)':<40} {'95.47%':>10}")
print(f"{'Ours v3 (53% coverage, lemma)':<40} {test_acc*100:>9.2f}%")
print(f"\nBest dev accuracy : {best_acc*100:.2f}%")
print(f"Model saved to    : best_model_v3.pt")
