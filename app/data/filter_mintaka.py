import json

with open("app/data/mintaka.json", encoding="utf-8") as f:
    data = json.load(f)

filtered = [item for item in data if item.get("language") == "de"]

with open("app/data/mintaka_filtered.json", "w", encoding="utf-8") as f:
    json.dump(filtered, f, ensure_ascii=False, indent=2)
with open("app/data/mintaka.json", encoding="utf-8") as f:
    data = json.load(f)

with open("app/data/mintaka_filtered.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
