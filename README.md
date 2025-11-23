# Medical RAG Assistant

A specialized medical chatbot designed to provide accurate information about drug interactions, food interactions, and general medication safety. This system leverages a Retrieval-Augmented Generation (RAG) architecture, combining a custom-tuned Large Language Model (LLM) with a local vector database and real-time web search to deliver reliable and context-aware responses
the vector DB Folder :https://drive.google.com/drive/folders/1tSkeSgLcuT7-5idryCxOsEBI7FL7x5Kh?usp=sharing

## 🚀 Features

-   **Specialized Medical Knowledge:** Expertly handles queries regarding:
    -   Drug-Drug Interactions (DDI)
    -   Drug-Food Interactions
    -   General Drug Information & Safety
-   **Hybrid Retrieval Engine:**
    -   **Local Knowledge Base:** Uses FAISS vector stores for fast, semantic retrieval of medical data.
    -   **Web Search Integration:** Augments local data with Google Custom Search for the latest information.
-   **Smart Query Routing:**
    -   Automatically detects casual conversation (greetings, identity questions) and responds instantly with static messages.
    -   Routes medical queries to the RAG pipeline for deep analysis.
-   **Safety-First Design:**
    -   Prioritizes patient safety with strict system prompts.
    -   Provides clear disclaimers and avoids generating dangerous treatment recommendations.
-   **Context-Aware Conversations:** Maintains chat history to understand follow-up questions (e.g., "Why?", "Tell me more").
-   **User-Friendly Interface:** Built with Gradio for a clean, responsive web-based chat experience.

## 🛠️ Tech Stack

-   **Language:** Python 3.8+
-   **LLM:** `Omnia118/qwen0.5b_ddi` (Fine-tuned Qwen model)
-   **Orchestration:** LangChain
-   **Vector Database:** FAISS (Facebook AI Similarity Search)
-   **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
-   **Web Search:** Google Custom Search API
-   **Interface:** Gradio
-   **ML Framework:** PyTorch, Transformers

## 📂 Project Structure

```
d:\chatbot\rag_implmetation\
├── config\                 # Configuration settings
│   └── config.py          # Central config class (paths, model names, API keys)
├── core\                   # Core application logic
│   ├── chat_manager.py    # Manages chat history and context
│   ├── rag_service.py     # Main RAG pipeline and LLM generation
│   └── retrieval.py       # FAISS retrieval engine logic
├── faiss_indices\          # Pre-built vector databases
│   ├── faiss_drugs/       # General drug info index
│   ├── faiss_drugdrug/    # Drug-drug interaction index
│   └── faiss_food/        # Drug-food interaction index
├── ui\                     # User Interface
│   └── gradio_app.py      # Gradio layout and event handlers
├── utils\                  # Utility functions
│   ├── helpers.py         # Web search and query detection logic
│   └── __init__.py        # Exports
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## ⚙️ Installation

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd rag_implmetation
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python -m venv .venv
    # Activate on Windows:
    .venv\Scripts\activate
    # Activate on Linux/Mac:
    source .venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🔧 Configuration

The application comes with default configuration in `config/config.py`. You can override these settings using environment variables or by modifying the file directly.

**Key Configuration Options:**

-   `LLM_MODEL`: The Hugging Face model ID (Default: `Omnia118/qwen0.5b_ddi`).
-   `DEVICE`: `cuda` (GPU) or `cpu`. Automatically detected.
-   `GOOGLE_API_KEY`: Your Google Custom Search API key.
-   `GOOGLE_SEARCH_ENGINE_ID`: Your Google Search Engine ID (CX).
-   `GRADIO_SERVER_PORT`: Port for the UI (Default: `7000`).

## ▶️ Usage

1.  **Start the Application:**
    ```bash
    python main.py
    ```

2.  **Access the Interface:**
    -   The application will launch a local server, typically at `http://127.0.0.1:7000`.
    -   A public Gradio link (shareable) may also be generated if configured.

3.  **Interact:**
    -   **Casual:** Say "Hi" or "Who are you?" to test the static response system.
    -   **Medical:** Ask questions like:
        -   "Can I take ibuprofen with paracetamol?"
        -   "What are the food interactions for warfarin?"
        -   "Side effects of metformin?"

## ⚠️ Medical Disclaimer

**This AI assistant is for informational purposes only.**

-   It is **NOT** a substitute for professional medical advice, diagnosis, or treatment.
-   Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
-   Never disregard professional medical advice or delay in seeking it because of something you have read from this system.
-   In case of a medical emergency, call your doctor or emergency services immediately.
