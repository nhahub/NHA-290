from config import Config
from core import ChatManager, RAGService, RetrievalEngine
from utils import web_search_google, is_medical_query
from ui.gradio_app import create_gradio_interface
import uuid
from typing import List, Tuple

def initialize_system():
    print("="*60)
    print("Initializing Medical RAG Assistant")
    print("="*60)
    
    config = Config.from_env()
    print("Configuration loaded")
    
    chat_manager = ChatManager(max_history=config.MAX_HISTORY_LENGTH)
    print("Chat Manager initialized")
    
    rag_service = RAGService(config.LLM_MODEL, config.DEVICE)
    print("RAG Service initialized")
    
    retrieval_engine = RetrievalEngine(config.FAISS_ROOT, config.EMBEDDING_MODEL)
    print("Retrieval Engine initialized")
    
    print("="*60)
    print("System ready!")
    print("="*60)
    
    return config, chat_manager, rag_service, retrieval_engine

def main():
    config, chat_manager, rag_service, retrieval_engine = initialize_system()

    # Answer function
    def answer_question(query: str, session_id: str, history: List[dict]) -> Tuple[str, List[dict]]:
        if not query.strip():
            return "Please ask a question.", history
        
        chat_history = chat_manager.get_history(session_id, format_for_llm=True)

        # Check if it's a greeting or identity question - handle with casual response
        query_lower = query.lower().strip()
        
        from utils.helpers import GREETING_PATTERNS, IDENTITY_PATTERNS
        
        is_greeting = any(
            query_lower == pattern or 
            query_lower.startswith(pattern + " ") or
            query_lower.startswith(pattern + "!")
            for pattern in GREETING_PATTERNS
        )
        
        is_identity = any(pattern in query_lower for pattern in IDENTITY_PATTERNS)
        
        # Route greetings and identity questions to casual handler
        if is_greeting or is_identity:
            answer = rag_service.generate_casual(
                query=query,
                chat_history=chat_history,
                max_tokens=100,
                temperature=config.TEMPERATURE
            )
        elif not is_medical_query(query, has_chat_history=len(chat_history) > 0):
            # If not a greeting/identity and not a medical query, return static message
            answer = "I am a medical assistant specialized in drug interactions and medical information. Please ask me a medical question, or feel free to say hello!"
        else:
            # All other questions go through RAG - let LLM decide if it's medical
            docs_with_scores = retrieval_engine.retrieve(query, k=config.TOP_K_RETRIEVAL)
            
            # Filter by threshold 0.4
            filtered_docs = [doc for doc, score in docs_with_scores if score >= 0.4]
            
            if filtered_docs:
                context_text = "\n\n".join([f"[Relevance: {score:.2f}]\n{doc.page_content[:500]}" 
                                             for doc, score in docs_with_scores if score >= 0.4])
            else:
                context_text = "" # Empty context if no results pass threshold

            # Use Google Custom Search
            web_results = []
            if config.GOOGLE_API_KEY and config.GOOGLE_SEARCH_ENGINE_ID:
                web_results = web_search_google(query, config.GOOGLE_API_KEY, config.GOOGLE_SEARCH_ENGINE_ID, num_results=3)
            
            answer = rag_service.generate_with_context(
                query=query,
                context=context_text,
                web_results_raw=web_results,
                chat_history=chat_history,
                max_tokens=config.MAX_NEW_TOKENS,
                temperature=config.TEMPERATURE
            )
        
        
        chat_manager.add_message(session_id, "user", query)
        chat_manager.add_message(session_id, "assistant", answer)
        
        return answer, history  

    def clear_chat(session_id: str):
        chat_manager.clear_session(session_id)
        return [], str(uuid.uuid4())

    def export_chat(session_id: str):
        return chat_manager.export_session(session_id)

    demo = create_gradio_interface(
        answer_fn=answer_question,
        clear_fn=clear_chat,
        export_fn=export_chat,
        theme="soft"
    )

    print("\nLaunching Gradio interface...")
    demo.launch(
        server_port=config.GRADIO_SERVER_PORT,
        server_name=config.GRADIO_SERVER_NAME,
        share=config.GRADIO_SHARE
    )

if __name__ == "__main__":
    main()
