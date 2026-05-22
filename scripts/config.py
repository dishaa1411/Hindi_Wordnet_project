# config.py
# Central config file — all paths defined here
# Works on ANY computer automatically

import os

# Root of the project (one level up from scripts/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Folders
DATA_DIR    = os.path.join(BASE_DIR, "data")
DB_DIR      = os.path.join(BASE_DIR, "database")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Database files
DATA_TXT = os.path.join(DB_DIR, "data_txt")

# Output files
EDGES_TSV      = os.path.join(OUTPUTS_DIR, "wordnet_edges.tsv")
EDGES_CLEAN    = os.path.join(OUTPUTS_DIR, "wordnet_edges_clean.tsv")
EMB_PKL        = os.path.join(OUTPUTS_DIR, "hyperbolic_embeddings.pkl")
WORD_VEC_PKL   = os.path.join(OUTPUTS_DIR, "word_vectors_light.pkl")
BEST_MODEL     = os.path.join(OUTPUTS_DIR, "best_model.pt")
BEST_MODEL_V2  = os.path.join(OUTPUTS_DIR, "best_model_v2.pt")

# Treebank files
TRAIN_CONLLU = os.path.join(DATA_DIR, "hi_hdtb-ud-train.conllu")
DEV_CONLLU   = os.path.join(DATA_DIR, "hi_hdtb-ud-dev.conllu")
TEST_CONLLU  = os.path.join(DATA_DIR, "hi_hdtb-ud-test.conllu")

# Quick check — print paths on import if run directly
if __name__ == "__main__":
    print(f"BASE_DIR    : {BASE_DIR}")
    print(f"DATA_DIR    : {DATA_DIR}")
    print(f"DB_DIR      : {DB_DIR}")
    print(f"OUTPUTS_DIR : {OUTPUTS_DIR}")
    print()
    print("Checking all paths exist:")
    all_paths = {
        "data_txt"      : DATA_TXT,
        "edges_clean"   : EDGES_CLEAN,
        "embeddings"    : EMB_PKL,
        "word_vectors"  : WORD_VEC_PKL,
        "train_conllu"  : TRAIN_CONLLU,
        "dev_conllu"    : DEV_CONLLU,
        "test_conllu"   : TEST_CONLLU,
    }
    for name, path in all_paths.items():
        status = "✅" if os.path.exists(path) else "❌ NOT FOUND"
        print(f"  {name:<20} {status}")