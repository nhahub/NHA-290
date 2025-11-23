import os
import re
import json
from typing import List, Dict, Tuple, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class RetrievalEngine:
    """Optimized retrieval engine with synonym expansion and intent detection"""
    
    def __init__(self, faiss_root: str, embedding_model: str):
        self.faiss_root = faiss_root
        self.embedding_model = embedding_model
        self.stores: Dict[str, FAISS] = {}
        self.synonym_map: Dict[str, List[str]] = {}
        
        self._load_stores()
        self._build_synonym_map()
    
    def _load_stores(self):
        """Load all FAISS indices"""
        indices_config = {
            "general": os.path.join(self.faiss_root, "faiss_drugs"),
            "ddi": os.path.join(self.faiss_root, "faiss_drugdrug"),
            "food": os.path.join(self.faiss_root, "faiss_food")
        }
        
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        for key, path in indices_config.items():
            if os.path.isdir(path):
                try:
                    self.stores[key] = FAISS.load_local(
                        path,
                        embeddings,
                        allow_dangerous_deserialization=True
                    )
                    print(f"Loaded {key} index from {path}")
                except Exception as e:
                    print(f"Failed to load {key}: {e}")
            else:
                print(f"Index not found: {path}")
    
    def _build_synonym_map(self):
        """Build synonym map from cached documents"""
        cache_path = os.path.join(self.faiss_root, "faiss_drugs", "all_docs_cache.json")
        
        if not os.path.exists(cache_path):
            print("No synonym cache found")
            return
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            for item in cached_data:
                name = item.get("name", "").lower().strip()
                synonyms = item.get("synonyms", [])
                
                if isinstance(synonyms, str):
                    synonyms = [s.strip() for s in synonyms.split(";") if s.strip()]
                
                aliases = {name} | {s.lower().strip() for s in synonyms if s}
                aliases_list = sorted(list(aliases))
                
                for alias in aliases_list:
                    self.synonym_map[alias] = aliases_list
            
            print(f"Built synonym map with {len(self.synonym_map)} entries")
        
        except Exception as e:
            print(f"Error building synonym map: {e}")
    
    def detect_intent(self, query: str) -> str:
        """Detect query intent: general, ddi, or food"""
        q = query.lower()
        
        if any(w in q for w in ["food", "eat", "drink", "alcohol", "meal", "grapefruit", "vitamin"]):
            return "food"
        
        if any(w in q for w in ["interact", "interaction", "combine", "together", "between", "contraindication"]):
            return "ddi"
        
        pair_pattern = r'\b([a-z]+)\s+(?:and|with|\+|vs)\s+([a-z]+)\b'
        if re.search(pair_pattern, q):
            return "ddi"
        
        return "general"
    
    def expand_query(self, query: str) -> str:
        """Expand query with synonyms"""
        words = re.findall(r'\b[a-z][a-z0-9\-]+\b', query.lower())
        
        expanded_terms = set()
        for word in words:
            if word in self.synonym_map:
                expanded_terms.update(self.synonym_map[word][:3])
        
        if expanded_terms:
            return f"{query} {' '.join(expanded_terms)}"
        
        return query
    
    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Retrieve relevant documents with scores"""
        intent = self.detect_intent(query)
        expanded_query = self.expand_query(query)
        
        store_key = "general"
        if intent == "food" and "food" in self.stores:
            store_key = "food"
        elif intent == "ddi" and "ddi" in self.stores:
            store_key = "ddi"
        
        try:
            results = self.stores[store_key].similarity_search_with_score(
                expanded_query,
                k=k * 2
            )
            
            results_with_sim = [
                (doc, max(0.0, min(1.0, 1.0 - dist)))
                for doc, dist in results
            ]
            
            if store_key != "general" and "general" in self.stores:
                general_results = self.stores["general"].similarity_search_with_score(
                    expanded_query,
                    k=k
                )
                results_with_sim.extend([
                    (doc, max(0.0, min(1.0, 1.0 - dist)))
                    for doc, dist in general_results
                ])
            
            seen = set()
            deduped = []
            for doc, score in results_with_sim:
                key = (doc.metadata.get("drugbank_id"), doc.page_content[:100])
                if key not in seen:
                    seen.add(key)
                    deduped.append((doc, score))
            
            deduped.sort(key=lambda x: x[1], reverse=True)
            return deduped[:k]
        
        except Exception as e:
            print(f"Retrieval error: {e}")
            return []