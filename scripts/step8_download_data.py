# step8_download_data.py
# Goal: download the Hindi HDTB treebank automatically

from datasets import load_dataset

print("Downloading Hindi HDTB treebank...")
print("This may take a minute...")

dataset = load_dataset("universal_dependencies", "hi_hdtb")

print()
print(f"Train sentences : {len(dataset['train'])}")
print(f"Dev sentences   : {len(dataset['validation'])}")
print(f"Test sentences  : {len(dataset['test'])}")
print()
print("Sample sentence:")
print(dataset['train'][0])