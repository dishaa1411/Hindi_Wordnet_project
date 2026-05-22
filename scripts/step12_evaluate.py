# step12_evaluate.py
# Goal: evaluate our model on the test set AND
# compare against a baseline (without hyperbolic embeddings)
# This comparison is the key result of our paper!

import torch
import torch.nn as nn
import pickle
import conllu
from torchcrf import CRF
from collections import Counter
from torch.utils.data import Dataset, DataLoader

# ---- Load everything ----
print("Loading data...")

emb_path = r"C:\Users\gupta\Downloads\HindiWN_1_5\Hindi_Wordnet_project\outputs\hyperbolic_embeddings.pkl"
with open(emb_path, "rb") as f:
    emb_data = pickle.load(f)

embeddings_matrix = emb_data["embeddings"]
node_to_id        = emb_data["node_to_id"]

train_path = r"C:\Users\gupta\Downloads\HindiWN_1_5\Hindi_Wordnet_project\data\hi_hdtb-ud-train.conllu"
dev_path   = r"C:\Users\gupta\Downloads\HindiWN_1_5\Hindi_Wordnet_project\data\hi_hdtb-ud-dev.conllu"
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
dev_data   = load_conllu(dev_path)
test_data  = load_conllu(test_path)

# Build vocabularies
all_words  = [word for sent in train_data for word, pos in sent]
word_counts = Counter(all_words)
word_to_id  = {"<PAD>": 0, "<UNK>": 1}
for word, count in word_counts.items():
    if count >= 2:
        word_to_id[word] = len(word_to_id)

all_tags   = [pos for sent in train_data for word, pos in sent]
tag_list   = sorted(set(all_tags))
tag_to_id  = {tag: i for i, tag in enumerate(tag_list)}
id_to_tag  = {i: tag for tag, i in tag_to_id.items()}

# Dataset
class POSDataset(Dataset):
    def __init__(self, sentences, word_to_id, tag_to_id):
        self.sentences  = sentences
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
    word_seqs, tag_seqs = zip(*batch)
    max_len  = max(len(s) for s in word_seqs)
    word_pad = torch.zeros(len(batch), max_len, dtype=torch.long)
    tag_pad  = torch.zeros(len(batch), max_len, dtype=torch.long)
    mask     = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, (w, t) in enumerate(zip(word_seqs, tag_seqs)):
        word_pad[i, :len(w)] = w
        tag_pad[i,  :len(t)] = t
        mask[i,     :len(w)] = True
    return word_pad, tag_pad, mask

test_dataset = POSDataset(test_data, word_to_id, tag_to_id)
test_loader  = DataLoader(test_dataset, batch_size=32,
                          shuffle=False, collate_fn=collate_fn)

# ---- Model definition ----
HYP_DIM    = 10
VOCAB_SIZE = len(word_to_id)
NUM_TAGS   = len(tag_to_id)
EMB_DIM    = 64
HIDDEN_DIM = 128

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
        word_embs = self.dropout(self.word_embedding(word_ids))
        combined  = torch.cat([word_embs, hyp_embs], dim=-1)
        lstm_out, _ = self.lstm(combined)
        emissions = self.linear(self.dropout(lstm_out))
        return emissions

    def predict(self, emissions, mask):
        return self.crf.decode(emissions, mask=mask)

# Baseline model: same but WITHOUT hyperbolic embeddings
class BiLSTMCRF_Baseline(nn.Module):
    def __init__(self, vocab_size, num_tags, emb_dim, hidden_dim):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size    = emb_dim,   # no hyperbolic embeddings!
            hidden_size   = hidden_dim,
            num_layers    = 2,
            batch_first   = True,
            bidirectional = True,
            dropout       = 0.3
        )
        self.linear  = nn.Linear(hidden_dim * 2, num_tags)
        self.crf     = CRF(num_tags, batch_first=True)
        self.dropout = nn.Dropout(0.3)

    def forward(self, word_ids, mask):
        word_embs   = self.dropout(self.word_embedding(word_ids))
        lstm_out, _ = self.lstm(word_embs)
        emissions   = self.linear(self.dropout(lstm_out))
        return emissions

    def loss(self, emissions, tags, mask):
        return -self.crf(emissions, tags, mask=mask, reduction="mean")

    def predict(self, emissions, mask):
        return self.crf.decode(emissions, mask=mask)

# ---- Hyperbolic embedding lookup ----
def get_hyp_embeddings(word_ids_batch, word_to_id,
                        node_to_id, embeddings_matrix):
    batch_size, seq_len = word_ids_batch.shape
    hyp_embs  = torch.zeros(batch_size, seq_len, HYP_DIM)
    id_to_word = {v: k for k, v in word_to_id.items()}
    for i in range(batch_size):
        for j in range(seq_len):
            word    = id_to_word.get(word_ids_batch[i, j].item(), "<UNK>")
            node_id = node_to_id.get(word, None)
            if node_id is not None:
                hyp_embs[i, j] = embeddings_matrix[node_id]
    return hyp_embs

# ---- Evaluate function with per-tag breakdown ----
def evaluate_detailed(model, loader, use_hyperbolic=True):
    model.eval()
    per_tag_correct = Counter()
    per_tag_total   = Counter()

    with torch.no_grad():
        for word_ids, tag_ids, mask in loader:
            if use_hyperbolic:
                hyp_embs  = get_hyp_embeddings(word_ids, word_to_id,
                                               node_to_id, embeddings_matrix)
                emissions = model(word_ids, hyp_embs, mask)
            else:
                emissions = model(word_ids, mask)

            preds = model.predict(emissions, mask)

            for i, pred_seq in enumerate(preds):
                true_seq = tag_ids[i, :len(pred_seq)].tolist()
                for p, t in zip(pred_seq, true_seq):
                    tag_name = id_to_tag[t]
                    per_tag_total[tag_name]   += 1
                    if p == t:
                        per_tag_correct[tag_name] += 1

    overall = sum(per_tag_correct.values()) / sum(per_tag_total.values())
    return overall, per_tag_correct, per_tag_total

# ---- Load our trained model ----
print("Loading trained model...")
model_path = r"C:\Users\gupta\Downloads\HindiWN_1_5\Hindi_Wordnet_project\outputs\best_model.pt"
our_model  = BiLSTMCRF(VOCAB_SIZE, NUM_TAGS, EMB_DIM, HIDDEN_DIM, HYP_DIM)
our_model.load_state_dict(torch.load(model_path, weights_only=True))

# ---- Train baseline model ----
print("Training baseline model (without hyperbolic embeddings)...")
print("This will take a few minutes...\n")

baseline   = BiLSTMCRF_Baseline(VOCAB_SIZE, NUM_TAGS, EMB_DIM, HIDDEN_DIM)
optimizer  = torch.optim.Adam(baseline.parameters(), lr=0.001)

train_dataset = POSDataset(train_data, word_to_id, tag_to_id)
train_loader  = DataLoader(train_dataset, batch_size=32,
                           shuffle=True, collate_fn=collate_fn)

for epoch in range(15):
    baseline.train()
    total_loss  = 0
    num_batches = 0
    for word_ids, tag_ids, mask in train_loader:
        emissions = baseline(word_ids, mask)
        loss      = baseline.loss(emissions, tag_ids, mask)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(baseline.parameters(), 5.0)
        optimizer.step()
        total_loss  += loss.item()
        num_batches += 1
    if (epoch + 1) % 5 == 0:
        print(f"  Baseline Epoch {epoch+1}/15 | Loss: {total_loss/num_batches:.4f}")

# ---- Final evaluation ----
print("\n" + "="*55)
print("FINAL RESULTS ON TEST SET")
print("="*55)

our_acc, our_correct, our_total = evaluate_detailed(
    our_model, test_loader, use_hyperbolic=True)
base_acc, base_correct, base_total = evaluate_detailed(
    baseline, test_loader, use_hyperbolic=False)

print(f"\n{'Model':<35} {'Accuracy':>10}")
print("-"*55)
print(f"{'Baseline BiLSTM-CRF':<35} {base_acc*100:>9.2f}%")
print(f"{'Our model (+ hyperbolic emb)':<35} {our_acc*100:>9.2f}%")
print(f"{'Improvement':<35} {(our_acc-base_acc)*100:>+9.2f}%")

print(f"\n\nPer-tag breakdown:")
print(f"{'Tag':<10} {'Baseline':>10} {'Ours':>10} {'Diff':>10}")
print("-"*45)
for tag in sorted(our_total.keys()):
    our_tag_acc  = our_correct[tag]  / our_total[tag]
    base_tag_acc = base_correct[tag] / base_total[tag]
    diff         = our_tag_acc - base_tag_acc
    print(f"{tag:<10} {base_tag_acc*100:>9.2f}% {our_tag_acc*100:>9.2f}% {diff*100:>+9.2f}%")
