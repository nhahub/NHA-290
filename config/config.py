import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """Central configuration for Medical RAG system"""
    
    # Model Configuration
    LLM_MODEL: str = "Omnia118/qwen0.5b_ddi"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # FAISS Indices Paths
    FAISS_ROOT: str = "faiss_indices"
    FAISS_DRUGS: str = os.path.join(FAISS_ROOT, "faiss_drugs")
    FAISS_DDI: str = os.path.join(FAISS_ROOT, "faiss_drugdrug")
    FAISS_FOOD: str = os.path.join(FAISS_ROOT, "faiss_food")
    
    # Google Search API
    GOOGLE_API_KEY: Optional[str] = "AIzaSyB4A_1Tz4TvGa8oS0kpPba6V-B5Hd9hgX8"
    GOOGLE_SEARCH_ENGINE_ID: Optional[str] = "f12239d1c79a2404c"
    
    # Generation Parameters
    MAX_NEW_TOKENS: int = 300
    TEMPERATURE: float = 0.7
    TOP_K_RETRIEVAL: int = 5
    
    # Chat History
    MAX_HISTORY_LENGTH: int = 10
    
    # Device
    DEVICE: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    
    # UI Configuration
    GRADIO_SERVER_PORT: int = 7000
    GRADIO_SERVER_NAME: str = "127.0.0.1"
    GRADIO_SHARE: bool = True

    @classmethod
    def from_env(cls):
        """Load config from environment variables with fallbacks"""
        return cls(
            GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY") or "AIzaSyB4A_1Tz4TvGa8oS0kpPba6V-B5Hd9hgX8",
            GOOGLE_SEARCH_ENGINE_ID=os.getenv("GOOGLE_SEARCH_ENGINE_ID") or "f12239d1c79a2404c",
            GRADIO_SHARE=os.getenv("GRADIO_SHARE", "True").lower() == "true"
        )

