"""RAG setup: FAISS vector store with Google embeddings and retriever tool.

This module builds a FAISS index from a local medical knowledge text file
and exposes a retriever suitable for LangChain tools.

Replace the sample dataset with SNOMED CT / DrugBank / Medline records by
loading and chunking those sources instead of the bundled text file.
"""

from __future__ import annotations

import os
from pathlib import Path
import zipfile
import os
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from .config import SETTINGS


DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_KNOWLEDGE_FILE = DATA_DIR / "medical_guidelines.txt"
DEFAULT_FAISS_INDEX = DATA_DIR / "faiss_medical_index"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZIP_FILE = REPO_ROOT / "CLEANED T&O.zip"

# Desktop CSV targets (user-specific)
USER_HOME = Path.home()
DESKTOP = USER_HOME / "Desktop"
CSV_CANDIDATES = [
    DESKTOP / "TwoSidesData.csv",
    DESKTOP / "OffSidesData.csv",
    DESKTOP / "TwoSidesData",
    DESKTOP / "OffSidesData",
]


def _load_documents_from_text(file_path: Path) -> list[Document]:
    """Load medical knowledge text into LangChain Document objects.

    Args:
        file_path: Path to a text file containing guidelines/knowledge.

    Returns:
        A list of Documents ready for splitting.
    """

    if not file_path.exists():
        # Create a minimal seed file if missing
        file_path.write_text(
            (
                "General medical guidance for education purposes only.\n\n"
                "Red flags: chest pain, severe shortness of breath, confusion,\n"
                "severe allergic reaction (anaphylaxis), uncontrolled bleeding.\n\n"
                "Self-care for common cold: rest, hydration, acetaminophen for fever,\n"
                "seek care if symptoms worsen or persist beyond 10 days.\n"
            ),
            encoding="utf-8",
        )

    text = file_path.read_text(encoding="utf-8")
    return [Document(page_content=text, metadata={"source": str(file_path)})]


def _is_textlike(name: str) -> bool:
    lower = name.lower()
    return lower.endswith((".txt", ".md", ".markdown", ".csv", ".tsv", ".json"))


def _load_documents_from_zip(zip_path: Path) -> list[Document]:
    """Load all text-like files from a zip archive into Documents.

    This allows you to bundle curated medical content (e.g., SNOMED extracts,
    DrugBank notes, Medline summaries). Non-text files are skipped.
    """
    docs: list[Document] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir() or not _is_textlike(info.filename):
                continue
            try:
                with zf.open(info, "r") as fh:
                    raw = fh.read()
                    try:
                        text = raw.decode("utf-8")
                    except UnicodeDecodeError:
                        # Fallback common encodings
                        for enc in ("utf-16", "latin-1"):
                            try:
                                text = raw.decode(enc)
                                break
                            except UnicodeDecodeError:
                                continue
                        else:
                            continue
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={"source": f"zip://{zip_path.name}/{info.filename}"},
                        )
                    )
            except Exception:
                # Skip problematic entries silently to keep indexing robust
                continue
    return docs


def _load_documents_from_csv_paths(paths: list[Path]) -> list[Document]:
    """Load CSV files as plain text documents.

    We avoid adding new dependencies; content is treated as text and chunked later.
    """
    docs: list[Document] = []
    for p in paths:
        if not p.exists() or p.is_dir():
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = p.read_text(encoding="latin-1")
            except Exception:
                continue
        docs.append(Document(page_content=text, metadata={"source": str(p)}))
    return docs


def _split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        length_function=len,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def build_or_load_retriever(
    knowledge_file: Path = DEFAULT_KNOWLEDGE_FILE,
    index_dir: Path = DEFAULT_FAISS_INDEX,
    zip_file: Path | None = DEFAULT_ZIP_FILE,
) -> FAISS:
    """Build or load a FAISS vector store retriever.

    If the FAISS index directory exists, it will be loaded. Otherwise, the
    index will be built from the provided knowledge file.

    Returns:
        A FAISS store (which can be used as a retriever via .as_retriever()).
    """

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=SETTINGS.google_api_key
    )

    if index_dir.exists():
        return FAISS.load_local(
            folder_path=str(index_dir), embeddings=embeddings, allow_dangerous_deserialization=True
        )

    documents: list[Document] = []
    # 1) Prefer Desktop CSVs if present
    csv_existing = [p for p in CSV_CANDIDATES if p.exists() and not p.is_dir()]
    if csv_existing:
        documents = _load_documents_from_csv_paths(csv_existing)
    # 2) Else prefer ZIP dataset if present
    if not documents and zip_file and zip_file.exists():
        documents = _load_documents_from_zip(zip_file)
    # Fallback to single text file
    if not documents:
        documents = _load_documents_from_text(knowledge_file)
    chunks = _split_documents(documents)
    store = FAISS.from_documents(chunks, embedding=embeddings)
    store.save_local(folder_path=str(index_dir))
    return store


# Global FAISS store and retriever instance
FAISS_STORE: Optional[FAISS] = None
RETRIEVER = None


def get_retriever():
    """Return a shared retriever instance, building the index on first use."""
    global FAISS_STORE, RETRIEVER
    if FAISS_STORE is None:
        FAISS_STORE = build_or_load_retriever()
        RETRIEVER = FAISS_STORE.as_retriever(search_kwargs={"k": 4})
    return RETRIEVER


