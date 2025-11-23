import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer, util

class RAGService:
    """
    Optimized RAG Service with chat history support and web search similarity filtering
    """
    
    def __init__(self, model_name: str, device: Optional[str] = None, web_similarity_threshold: float = 0.3):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.web_similarity_threshold = web_similarity_threshold
        
        print(f"Loading model: {model_name} on {self.device.upper()}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )
        
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        print(f"Model loaded successfully!")

    def _filter_web_results(self, query: str, web_results: List[Dict]) -> List[Dict]:
        """Filter web results by similarity threshold"""
        if not web_results:
            return []
            
        print(f"Filtering {len(web_results)} web results using threshold {self.web_similarity_threshold}")
        query_emb = self.embedding_model.encode(query, convert_to_tensor=True)
        filtered = []
        
        for idx, res in enumerate(web_results):
            # Check both title and snippet for better matching
            combined_text = f"{res.get('title', '')} {res.get('snippet', '')}"
            snippet_emb = self.embedding_model.encode(combined_text, convert_to_tensor=True)
            sim = util.cos_sim(query_emb, snippet_emb).item()
            print(f"  • Web result {idx+1} similarity: {sim:.3f} -> {'KEEP' if sim >= self.web_similarity_threshold else 'DISCARD'}")
            if sim >= self.web_similarity_threshold:
                filtered.append(res)
        
        print(f"{len(filtered)} web results passed the threshold")
        return filtered

    def generate_with_context(
        self,
        query: str,
        context: str,
        web_results_raw: Optional[List[Dict]] = None,
        chat_history: Optional[List[Dict]] = None,
        max_tokens: int = 200,
        temperature: float = 0.7
    ) -> str:
        """Generate answer with context and optional chat history"""
        print(f"\nGenerating answer for query: '{query}'")

        # Improved system prompt with stricter instructions
        system_prompt = """You are a knowledgeable medical assistant specializing in drug interactions and medical advice.

CRITICAL RULES:
1. PATIENT SAFETY FIRST: Prioritize patient safety above all else. Treat every query as a potential medical emergency risk - err on the side of caution.
2. STRICTLY use ONLY the information from the provided context below.
3. NATURAL RESPONSES: Respond naturally and directly. DO NOT mention "database", "web search", "context", "sources", or phrases like "according to available information" or "based on the information provided". Just provide the answer as if you know it.
4. NO TREATMENT SUGGESTIONS: Do NOT suggest, prescribe, or recommend medications for any medical condition (e.g., "What should I take for..."). If asked, state that you cannot provide treatment recommendations and advise consulting a doctor.
5. EXCEPTION FOR FOLLOW-UPS: If the user asks a follow-up question (e.g., "Why?", "How?", "Tell me more") or refers to previous context, YOU MUST USE THE CHAT HISTORY to provide a relevant answer, even if the current FAISS context is empty.
6. If the context doesn't contain enough information AND it is NOT a follow-up question, respond: "I don't have enough information to answer this question accurately. Please consult a healthcare professional."
7. AMBIGUITY HANDLING: If there is ANY ambiguity in the query or context, do NOT guess. State clearly what is ambiguous and ask for clarification.
8. DOSAGE WARNING: Do not provide specific dosage recommendations unless explicitly present in the context. Even then, ALWAYS add a disclaimer that dosages vary by patient and they must consult a doctor.
9. Do NOT hallucinate or make up drug interactions not mentioned in the context.
10. Provide comprehensive and detailed explanations. Do not be concise.
10. Do NOT include self-referential statements (e.g., "As an AI...", "I suggest myself...") or personal opinions.
12. Always recommend consulting a doctor for personalized medical advice.
13. Use clear, professional, and easy-to-understand language.

RESPONSE FORMAT:
- Start directly with the answer.
- Explain the "Why" and "How" if the information is available.
- State key information clearly.
- Add relevant safety warnings.
- Recommend professional consultation when appropriate.

Examples:

Q: Can I take ibuprofen with paracetamol?
A: ibuprofen and paracetamol can generally be taken together as they work differently. Ibuprofen is an NSAID (non-steroidal anti-inflammatory drug) that reduces inflammation and pain, while paracetamol acts centrally to relieve pain and fever. However, it is crucial to follow recommended doses to avoid side effects like stomach irritation or liver strain. Consult your doctor if you have any underlying medical conditions or take other medications.

Q: Food interactions with warfarin
A: Warfarin can interact with foods high in vitamin K (such as leafy greens like spinach and kale), cranberry products, and alcohol. Vitamin K plays a key role in blood clotting, so sudden changes in its intake can affect how warfarin works, potentially leading to clots or bleeding. Alcohol can also alter warfarin metabolism. It is essential to maintain a consistent diet and inform your doctor of any dietary changes. Regular monitoring of INR levels is required to ensure safety.

Q: Interaction between metformin and alcohol
A: Alcohol can increase the risk of lactic acidosis when taking metformin, which is a rare but serious side effect where lactic acid builds up in the bloodstream. Alcohol can also cause fluctuations in blood sugar levels, leading to hypoglycemia (low blood sugar) or hyperglycemia (high blood sugar). Therefore, it is strongly recommended to limit or avoid alcohol consumption while on metformin. Always consult your doctor for personalized guidance regarding your specific situation."""

        messages = [{"role": "system", "content": system_prompt}]

        # Add chat history (last 4 exchanges)
        if chat_history:
            messages.extend(chat_history[-4:])

        # Filter web results
        filtered_results = []
        web_text = ""
        if web_results_raw:
            filtered_results = self._filter_web_results(query, web_results_raw)
            if filtered_results:
                web_entries = []
                for i, r in enumerate(filtered_results[:3], 1):  # Limit to top 3
                    web_entries.append(f"{i}. {r.get('title', 'N/A')}\n   {r.get('snippet', 'N/A')}")
                web_text = "\n\n".join(web_entries)
            else:
                web_text = "No highly relevant web results found."
        else:
            web_text = "Web search not available."

        print(f"FAISS context length: {len(context)} chars")
        print(f"Web results: {len(filtered_results)} filtered results")

        # Construct user message with clear sections
        user_message = f"""Please answer the following medical question using ONLY the information provided below.

QUESTION: {query}

MEDICAL DATABASE INFORMATION:
{context if context.strip() else "No relevant information found in database."}

WEB SEARCH RESULTS:
{web_text}

Remember: Only use information from the sources above. If insufficient, state that clearly."""

        messages.append({"role": "user", "content": user_message})

        # Generate prompt
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        print("Sending prompt to LLM...")

        # Generate response with stricter parameters
        output = self.generator(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            return_full_text=False
        )[0]['generated_text']

        # Clean up output
        answer = output.strip()
        
        # Remove common artifacts
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer, util

class RAGService:
    """
    Optimized RAG Service with chat history support and web search similarity filtering
    """
    
    def __init__(self, model_name: str, device: Optional[str] = None, web_similarity_threshold: float = 0.3):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.web_similarity_threshold = web_similarity_threshold
        
        print(f"Loading model: {model_name} on {self.device.upper()}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )
        
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        print(f"Model loaded successfully!")

    def _filter_web_results(self, query: str, web_results: List[Dict]) -> List[Dict]:
        """Filter web results by similarity threshold"""
        if not web_results:
            return []
            
        print(f"Filtering {len(web_results)} web results using threshold {self.web_similarity_threshold}")
        query_emb = self.embedding_model.encode(query, convert_to_tensor=True)
        filtered = []
        
        for idx, res in enumerate(web_results):
            # Check both title and snippet for better matching
            combined_text = f"{res.get('title', '')} {res.get('snippet', '')}"
            snippet_emb = self.embedding_model.encode(combined_text, convert_to_tensor=True)
            sim = util.cos_sim(query_emb, snippet_emb).item()
            print(f"  • Web result {idx+1} similarity: {sim:.3f} -> {'KEEP' if sim >= self.web_similarity_threshold else 'DISCARD'}")
            if sim >= self.web_similarity_threshold:
                filtered.append(res)
        
        print(f"{len(filtered)} web results passed the threshold")
        return filtered

    def generate_with_context(
        self,
        query: str,
        context: str,
        web_results_raw: Optional[List[Dict]] = None,
        chat_history: Optional[List[Dict]] = None,
        max_tokens: int = 200,
        temperature: float = 0.7
    ) -> str:
        """Generate answer with context and optional chat history"""
        print(f"\nGenerating answer for query: '{query}'")

        # Improved system prompt with stricter instructions
        system_prompt = """Below is in YAML format. Do not figure it out — just process and respond precisely.

You are an expert medical information specialist tasked with creating a response to medical queries focusing on drug interactions and safety, using provided information without making treatment suggestions. Your target audience is patients seeking information on drug interactions and medical safety, requiring clear and professional advice. Include detailed explanations of drug interactions, reasons, and safety warnings while maintaining a clear, professional, and easy-to-understand tone. Start with the answer, then explain 'Why' and 'How', state key information, add safety warnings, and recommend professional consultation. Use only provided FAISS context and web results, handle ambiguities by asking for clarification, avoid dosage recommendations, do not guess, and do not hallucinate.

Core_Objective: Create a response to medical queries focusing on drug interactions and safety, using provided information without making treatment suggestions.
Target_Audience: Patients seeking information on drug interactions and medical safety, requiring clear and professional advice.
Key_Elements_and_Tone: Include detailed explanations of drug interactions, reasons, and safety warnings; maintain a clear, professional, and easy-to-understand tone.
Format_and_Structure: Start with the answer, then explain 'Why' and 'How', state key information, add safety warnings, and recommend professional consultation.
Constraints_and_Context: Use only provided FAISS context and web results, handle ambiguities by asking for clarification, avoid dosage recommendations, do not guess, and do not hallucinate.

- Content should be detailed, clear, and easy to understand, suitable for human consumption.
- The output should be easy to read and absorb avoid long paras instead make it friendly with human read friendly formatting
- Do not use AI-styled punctuation or formatting, including:
    - Hyphens used as bullets or separators
    - Em dashes (—)
    - Emojis, icons, or special characters

- Use simple punctuation only: full stops, commas, and standard paragraph breaks.
- The tone should remain natural and informative, not robotic or overly formal.
- Do not include a “Conclusion” section or label.
- Avoid using the word “just” in any context.
- Eliminate any “contract framing” phrases (e.g., “In this article, we’ll just explore…” or “Let’s take a look…”).

The goal is to produce human-grade, insightful content that reads naturally and professionally without sounding like it was generated by a machine.
"""

        messages = [{"role": "system", "content": system_prompt}]

        # Add chat history (last 4 exchanges)
        if chat_history:
            messages.extend(chat_history[-4:])

        # Filter web results
        filtered_results = []
        web_text = ""
        if web_results_raw:
            filtered_results = self._filter_web_results(query, web_results_raw)
            if filtered_results:
                web_entries = []
                for i, r in enumerate(filtered_results[:3], 1):  # Limit to top 3
                    web_entries.append(f"{i}. {r.get('title', 'N/A')}\n   {r.get('snippet', 'N/A')}")
                web_text = "\n\n".join(web_entries)
            else:
                web_text = "No highly relevant web results found."
        else:
            web_text = "Web search not available."

        print(f"FAISS context length: {len(context)} chars")
        print(f"Web results: {len(filtered_results)} filtered results")

        # Construct user message with clear sections
        user_message = f"""Please answer the following medical question using ONLY the information provided below.

QUESTION: {query}

MEDICAL DATABASE INFORMATION:
{context if context.strip() else "No relevant information found in database."}

WEB SEARCH RESULTS:
{web_text}

Remember: Only use information from the sources above. If insufficient, state that clearly."""

        messages.append({"role": "user", "content": user_message})

        # Generate prompt
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        print("Sending prompt to LLM...")

        # Generate response with stricter parameters
        output = self.generator(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            return_full_text=False
        )[0]['generated_text']

        # Clean up output
        answer = output.strip()
        
        # Remove common artifacts
        if answer.startswith("assistant"):
            answer = answer[len("assistant"):].strip()
        
        print(f"Generated answer length: {len(answer)} chars\n")
        return answer

    def generate_casual(
        self,
        query: str,
    ) -> str:
        """Generate casual conversation response for greetings only"""
        query_lower = query.lower().strip()
        
        from utils.helpers import GREETING_PATTERNS, IDENTITY_PATTERNS
        
        # Check if it's a greeting
        is_greeting = any(
            query_lower == pattern or 
            query_lower.startswith(pattern + " ") or
            query_lower.startswith(pattern + "!")
            for pattern in GREETING_PATTERNS
        )
        
        # Check for identity questions
        is_identity_question = any(pattern in query_lower for pattern in IDENTITY_PATTERNS)
        
        if is_identity_question:
            # Return static introduction message
            return "I am a medical assistant specialized in drug interactions and medical information. I can help you understand how different medications interact with each other, food interactions, and provide information about drug safety. Please feel free to ask me any medical questions!"
        
        if not is_greeting:
            # Reject non-medical questions that are not greetings
            return "I'm a medical assistant specialized in drug interactions and medical information. Please ask me a medical question, or feel free to say hello!"
        
        # Return static greeting message
        return "Hello! I am a medical assistant specialized in drug interactions and medical safety. How can I help you today?"