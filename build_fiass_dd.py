# ========================================
# FAISS Builder for Drug-Drug Interactions (Clean Version)
# ========================================
# ========================================

import os
import csv
import json
from collections import defaultdict
from typing import List, Dict, Set
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import torch

# ======== CONFIG ========
INPUT_CSV = "D:\rag_implmetation\csv_export\drug_drug_interactions.csv"  # change if needed
OUTPUT_DIR = "D:\rag_implmetation\faiss_drugdrug"
PREPROCESSED_JSON = os.path.join(OUTPUT_DIR, "preprocessed_drugdrug.json")
FAISS_DIR = os.path.join(OUTPUT_DIR, "faiss_drugdrug")

MAX_INTERACTIONS = 30
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_BATCH_SIZE = 128


# ======== STEP 1: Preprocess CSV ========
def preprocess_csv(input_csv: str, output_json: str, max_interactions: int = 30):
    print(f"Processing CSV: {input_csv}")
    
    if not os.path.exists(input_csv):
        print(f"CSV not found: {input_csv}")
        print("\nAvailable files in /kaggle/input/:")
        for root, dirs, files in os.walk("/kaggle/input/"):
            for file in files:
                print(f"   {os.path.join(root, file)}")
        raise FileNotFoundError(f"CSV not found: {input_csv}")
    
    pair_to_interactions: Dict[str, Dict] = {}
    seen_interactions: Dict[str, Set[int]] = defaultdict(set)
    row_count = 0

    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_count += 1
            if row_count % 10000 == 0:
                print(f"   Processed {row_count:,} rows...")
            
            id1 = row.get("drugbank_id", "").strip()
            n1 = row.get("name", "").strip()
            n2 = row.get("other_drug_name", "").strip()
            text = row.get("interaction_description", "").strip()
            
            if not text or not n1 or not n2:
                continue
            
            key_a, key_b = sorted([n1.lower(), n2.lower()])
            pair_key = f"{key_a}||{key_b}"

            if pair_key in pair_to_interactions and len(pair_to_interactions[pair_key]["interactions"]) >= max_interactions:
                continue
            
            h = hash(text.lower())
            if h in seen_interactions[pair_key]:
                continue
            seen_interactions[pair_key].add(h)

            if len(text) > 500:
                text = text[:500] + "..."

            if pair_key not in pair_to_interactions:
                pair_to_interactions[pair_key] = {
                    "id1": id1,
                    "name1": n1,
                    "name2": n2,
                    "interactions": []
                }
            
            pair_to_interactions[pair_key]["interactions"].append(text)

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(list(pair_to_interactions.values()), f, ensure_ascii=False, indent=2)
    
    print(f"Preprocessed {len(pair_to_interactions):,} drug pairs")
    print(f"Saved to: {output_json}")
    return len(pair_to_interactions)


# ======== STEP 2: Load Documents ========
def load_documents_from_json(json_file: str) -> List[Document]:
    print("Loading documents from JSON...")
    with open(json_file, "r", encoding="utf-8") as f:
        all_pairs = json.load(f)

    documents = []
    for item in all_pairs:
        drug1, drug2 = sorted([item["name1"], item["name2"]], key=str.lower)
        interactions_text = "\n".join(f"{i+1}. {txt}" for i, txt in enumerate(item["interactions"]))
        
        page_content = (
            f"Drug-Drug Interaction: {drug1} and {drug2}\n"
            f"Primary Drug: {drug1} (DrugBank ID: {item['id1']})\n"
            f"Interacting Drug: {drug2}\n"
            f"Total Interactions: {len(item['interactions'])}\n\nDetails:\n{interactions_text}"
        )
        
        if len(page_content) > 3000:
            page_content = page_content[:3000] + "\n... (truncated)"
        
        documents.append(Document(
            page_content=page_content,
            metadata={
                "doc_type": "drug_drug_interaction",
                "drugbank_id": item["id1"],
                "drug1_name": drug1,
                "drug2_name": drug2,
                "interaction_count": len(item["interactions"]),
                "aliases": [f"{drug1} - {drug2}", f"{drug2} - {drug1}"]
            }
        ))
    
    print(f"Loaded {len(documents):,} documents")
    return documents


# ======== STEP 3: Build FAISS ========
def build_and_save_faiss(documents: List[Document], persist_dir: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Building FAISS on: {device.upper()}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": EMBED_BATCH_SIZE
        }
    )

    print(f"Embedding {len(documents):,} documents...")
    import time
    start = time.time()
    
    vectorstore = FAISS.from_documents(documents, embedding=embeddings)
    
    elapsed = time.time() - start
    print(f"Embedding completed in {elapsed:.2f} seconds")
    print(f"Speed: {len(documents)/elapsed:.1f} docs/sec")

    os.makedirs(persist_dir, exist_ok=True)
    vectorstore.save_local(persist_dir)
    print(f"FAISS index saved to: {persist_dir}")


# ======== MAIN PIPELINE ========
def main():
    print("="*70)
    print("DRUG-DRUG INTERACTION FAISS BUILDER")
    print("="*70)
    
    print(f"Working directory: {OUTPUT_DIR}")
    print(f"Input CSV: {INPUT_CSV}\n")
    
    try:
        num_pairs = preprocess_csv(INPUT_CSV, PREPROCESSED_JSON, MAX_INTERACTIONS)
        documents = load_documents_from_json(PREPROCESSED_JSON)
        build_and_save_faiss(documents, FAISS_DIR)
        
        print("="*70)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Drug pairs processed: {num_pairs:,}")
        print(f"Documents created: {len(documents):,}")
        print(f"FAISS index saved in: {FAISS_DIR}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        print("\nTroubleshooting:")
        print("1. Check INPUT_CSV path is correct")
        print("2. Make sure dataset is added to notebook")
        print("3. Enable GPU in Settings (optional)")

if __name__ == "__main__":
    main()
