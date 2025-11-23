import os
import json
from langchain_core.documents import Document

# ====== PATHS ======
input_json = "preprocessed_drugdrug.json"   # change if your file name is different
output_dir = "faiss_indices/faiss_drugdrug"
cache_path = os.path.join(output_dir, "all_docs_cache.json")

# ====== LOAD ORIGINAL JSON ======
with open(input_json, "r", encoding="utf-8") as f:
    all_pairs = json.load(f)

# ====== BUILD DOCUMENT METADATA ======
documents = []
for item in all_pairs:
    meta = {
        "doc_type": "drug_drug_interaction", 
        "drug_1_id": item.get("id1"),
        "drug_1_name": item.get("name1"),
        "drug_2_name": item.get("name2"),
        "interaction_text": " ".join(item.get("interactions", [])),
    }
    documents.append(Document(page_content="", metadata=meta))

# ====== SAVE CACHE ======
os.makedirs(output_dir, exist_ok=True)
with open(cache_path, "w", encoding="utf-8") as f:
    json.dump([d.metadata for d in documents], f, ensure_ascii=False, indent=2)

print(f" Created all_docs_cache.json at: {cache_path}")
print(f"Total entries: {len(documents)}")
