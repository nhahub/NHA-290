import os
import csv
import argparse
import json
from typing import List, Dict, Set, Generator
from collections import defaultdict

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ========= MEMORY-EFFICIENT CSV LOADER =========
def load_csv_rows_generator(path: str) -> Generator[Dict[str, str], None, None]:
    """Generator to read CSV rows one at a time - memory efficient for large files"""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row

# ========== DOCUMENT CONSTRUCTORS =========
def make_drug_docs(path: str) -> List[Document]:
    """Process drug information CSV"""
    docs = []
    for row in load_csv_rows_generator(path):
        drug_id = row.get('drugbank_id', '').strip()
        name = row.get('name', '').strip()
        description = row.get('description', '').strip()
        synonyms = row.get('synonyms', '').strip()
        
        if not drug_id or not name:
            continue
        
        summary = (
            f"Drug: {name}\n"
            f"DrugBank ID: {drug_id}\n"
            f"Description: {description}\n"
            f"Synonyms: {synonyms}"
        )
        
        docs.append(Document(
            page_content=summary,
            metadata={
                'doc_type': 'drug',
                'drugbank_id': drug_id,
                'name': name,
                'synonyms': synonyms,
                'name_lower': name.lower()
            }
        ))
    
    return docs



def make_food_docs(path: str) -> List[Document]:
    """Process drug-food interaction CSV"""
    docs = []
    for row in load_csv_rows_generator(path):
        drug_id = row.get('drugbank_id', '').strip()
        name = row.get('name', '').strip()
        food_interaction = row.get('food_interaction', '').strip()
        
        if not drug_id or not name or not food_interaction:
            continue
        
        summary = (
            f"Drug-Food Interaction: {name}\n"
            f"DrugBank ID: {drug_id}\n"
            f"Food Interaction: {food_interaction}"
        )
        
        docs.append(Document(
            page_content=summary,
            metadata={
                'doc_type': 'drug_food_interaction',
                'drugbank_id': drug_id,
                'name': name,
                'food_interaction': food_interaction,
                'name_lower': name.lower()
            }
        ))
    
    return docs

# ========== FAISS BUILDER & SAVER =========
def build_faiss_from_documents(documents: List[Document], model_name: str) -> FAISS:
    """Build FAISS index from documents with specified embedding model"""
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.from_documents(documents, embedding=embeddings)

def save_faiss(store: FAISS, persist_dir: str) -> None:
    """Save FAISS index to disk"""
    os.makedirs(persist_dir, exist_ok=True)
    store.save_local(persist_dir)

def save_metadata(docs: List[Document], persist_dir: str) -> None:
    """Save document metadata cache as JSON"""
    os.makedirs(persist_dir, exist_ok=True)
    cache_path = os.path.join(persist_dir, "all_docs_cache.json")
    
    metadata_list = []
    for doc in docs:
        meta = doc.metadata.copy()
        if 'aliases' in meta and isinstance(meta['aliases'], list):
            meta['aliases'] = meta['aliases'][:2]  # Limit to save space
        metadata_list.append(meta)
    
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)

# ========== MAIN ==========
def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS indices from drug-related CSVs for RAG system"
    )
    parser.add_argument("--drugs", default="csv_export/drugs.csv",
                        help="Path to drugs CSV file")
    parser.add_argument("--food", default="csv_export/drug_food_interactions.csv",
                        help="Path to drug-food interactions CSV file")
    parser.add_argument("--model", 
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="HuggingFace embedding model name")
    parser.add_argument("--out", default="faiss_indices",
                        help="Output root directory for FAISS indices")
    args = parser.parse_args()

    dataset_configs = [
        ("drugs", make_drug_docs, args.drugs),
        ("food", make_food_docs, args.food)
    ]

    for kind, make_docs_fn, file_path in dataset_configs:
        print(f"\n{'='*60}")
        print(f"Processing: {kind.upper()}")
        print(f"Source file: {file_path}")
        print(f"{'='*60}")
        
        if not os.path.exists(file_path):
            print(f"WARNING: File not found, skipping: {file_path}\n")
            continue
        
        try:
            docs = make_docs_fn(file_path)
            print(f"Loaded {len(docs):,} documents")
            
            if not docs:
                print(f"No documents created, skipping index creation\n")
                continue
            
            persist_dir = os.path.join(args.out, f"faiss_{kind}")
            print(f"Building FAISS index at: {persist_dir}")
            
            store = build_faiss_from_documents(docs, args.model)
            save_faiss(store, persist_dir)
            save_metadata(docs, persist_dir)
            
            print(f"Successfully saved: {kind}\n")
            
        except Exception as e:
            print(f"ERROR processing {kind}: {str(e)}\n")
            continue

    print(f"\n{'='*60}")
    print("All processing complete!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()