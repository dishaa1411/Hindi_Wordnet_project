# step9_inspect_treebank.py
# Goal: understand what the HDTB treebank looks like

file_path = r"C:\Users\gupta\Downloads\HindiWN_1_5\Hindi_Wordnet_project\data\hi_hdtb-ud-train.conllu"

with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Print first 30 lines
print("First 30 lines of the treebank:\n")
for i, line in enumerate(lines[:30]):
    print(f"{i+1:3}: {line}", end="")

