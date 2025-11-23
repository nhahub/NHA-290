import re
from typing import List, Dict


# Common casual/general patterns
GREETING_PATTERNS = [
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "how are you", "what's up", "how do you do", "greetings"
]

IDENTITY_PATTERNS = [
    "who are you", "what are you", "who r u", "introduce yourself", 
    "what is your name"
]

CASUAL_PATTERNS = GREETING_PATTERNS + IDENTITY_PATTERNS + [
    "thanks", "thank you", "thx",
    "bye", "goodbye", "ok", "okay", "yes", "no", "cool", "nice",
    "tell me a joke",
    "weather", "news", "sport", "game", "movie", "music"
]

def is_medical_query(query: str, has_chat_history: bool = False) -> bool:
    """
    Detect if query is medical-related using keyword matching and heuristics.
    Returns True if the query appears to be medical, False otherwise.
    
    Args:
        query: The user's query string
        has_chat_history: Whether there is existing chat history (for follow-up detection)
    """
    query_lower = query.lower().strip()
    
    # Common follow-up question patterns - these should be treated as medical if there's chat history
    follow_up_patterns = [
        "why", "how", "why not", "how so", "explain", "tell me more", 
        "what do you mean", "can you elaborate", "more details", "more info",
        "what about", "and", "but", "also", "additionally"
    ]
    
    # If there's chat history and this looks like a follow-up, treat it as medical
    if has_chat_history:
        for pattern in follow_up_patterns:
            if query_lower == pattern or query_lower.startswith(pattern + " ") or query_lower.startswith(pattern + "?"):
                return True
    
    # Too short to be meaningful medical query usually (unless it's a follow-up)
    if len(query_lower) < 3:
        return False
    
    # Check for exact matches or starts with casual patterns
    for pattern in CASUAL_PATTERNS:
        if query_lower == pattern or query_lower.startswith(pattern + " ") or query_lower.startswith(pattern + "!"):
            return False
            
    # Comprehensive list of medical keywords
    medical_keywords = [
        # General terms
        "drug", "medication", "medicine", "pill", "tablet", "capsule", "syrup",
        "interaction", "side effect", "adverse", "reaction", "contraindication",
        "dosage", "dose", "prescription", "overdose", "toxicity",
        "treatment", "therapy", "cure", "remedy", "prevention",
        "symptom", "sign", "diagnosis", "prognosis",
        "doctor", "physician", "hospital", "clinic", "patient", "nurse",
        "health", "medical", "clinical", "pharmacy", "pharmacist",
        
        # Conditions & Diseases
        "disease", "disorder", "syndrome", "infection", "inflammation",
        "pain", "ache", "fever", "cough", "flu", "cold",
        "diabetes", "cancer", "tumor", "heart", "cardio", "blood pressure",
        "liver", "kidney", "renal", "hepatic", "lung", "respiratory",
        "brain", "neuro", "mental", "anxiety", "depression",
        "stomach", "gastric", "bowel", "intestine", "digestive",
        "skin", "derma", "rash", "allergy", "allergic","interaction",

        # Drug classes & specific common drugs (examples)
        "antibiotic", "antiviral", "antifungal", "painkiller", "analgesic",
        "aspirin", "paracetamol", "ibuprofen", "insulin", "metformin",
        "statin", "vitamin", "supplement", "vaccine","take","pregnancy","risk","use",
        
        # Biological
        "blood", "urine", "test", "exam", "surgery", "operation"
    ]
    
    # Check if any medical keyword is present
    has_medical_term = any(keyword in query_lower for keyword in medical_keywords)
    
    # Heuristic: If it has a medical term, it's likely medical.
    if has_medical_term:
        return True
        
    # If no keywords found, but query is long, it might be a description of symptoms.
    # However, to be safe and avoid answering general questions, we lean towards False if unsure.
    
    common_question_starters = ["how", "what", "can", "does", "is", "why", "when"]
    starts_with_question = any(query_lower.startswith(s) for s in common_question_starters)
    
    # If it starts with a question word but has no medical keywords, assume general.
    # Example: "What is the capital of France?" -> False
    
    return False

def web_search_google(query: str, api_key: str, engine_id: str, num_results: int = 3) -> List[Dict[str, str]]:
    """Perform Google Custom Search"""
    if not api_key or not engine_id:
        return []
    
    try:
        from googleapiclient.discovery import build
        
        service = build("customsearch", "v1", developerKey=api_key)
        result = service.cse().list(q=query, cx=engine_id, num=num_results).execute()
        
        return [
            {
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "link": item.get("link", "")
            }
            for item in result.get("items", [])
        ]
    except Exception as e:
        print(f"Web search error: {e}")
        return []
