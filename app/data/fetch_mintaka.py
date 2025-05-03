import json
from datasets import load_dataset

# Load the multilingual Mintaka QA split
ds = load_dataset("allenai/mintaka", "multilingual")
# Convert to list of dicts
all_qa = ds["train"].to_dicts()
# Write to file
with open('app/data/mintaka.json', 'w', encoding='utf-8') as f:
    json.dump(all_qa, f, ensure_ascii=False, indent=2)