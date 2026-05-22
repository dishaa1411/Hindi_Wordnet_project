# step2_parse.py
# Goal: parse every line of data_txt and extract:
#   - synset ID
#   - POS tag
#   - list of words
#   - definition
#   - relations to other synsets

from config import DATA_TXT
file_path = DATA_TXT

# POS code to human readable label
pos_map = {
    "01": "noun",
    "02": "adjective",
    "03": "verb",
    "04": "adverb"
}

# This will store all parsed synsets
synsets = []

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        
        # skip empty lines
        if not line:
            continue
        
        # split on the | separator
        # left side = data, right side = definition
        if "|" not in line:
            continue
            
        data_part, definition = line.split("|", 1)
        data_part = data_part.strip()
        definition = definition.strip()
        
        # split data part by spaces
        tokens = data_part.split()
        
        # first token = synset ID
        synset_id = tokens[0]
        
        # second token = POS code
        pos_code = tokens[1]
        pos_label = pos_map.get(pos_code, "unknown")
        
        # third token = number of words
        num_words = int(tokens[2])
        
        # next tokens are the words (joined by colon)
        words_token = tokens[3]
        words = words_token.split(":")
        
        # store it
        synsets.append({
            "id": synset_id,
            "pos": pos_label,
            "words": words,
            "definition": definition
        })

# Print summary
print(f"Total synsets parsed: {len(synsets)}")
print()

# Print first 3 as a sample
for s in synsets[:3]:
    print(f"ID: {s['id']}")
    print(f"POS: {s['pos']}")
    print(f"Words: {s['words']}")
    print(f"Definition: {s['definition']}")
    print("---")
