# Goal: peek inside the data_txt file and understand its structure

# This tells Python where your file is
file_path = r"C:\Users\gupta\Downloads\HindiWN_1_5\Hindi_Wordnet_project\database\data_txt"

# Open the file and read first 5 lines
with open(file_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        print(f"Line {i+1}:")
        print(line)
        print("---")
        if i == 4:  # stop after 5 lines
            break





