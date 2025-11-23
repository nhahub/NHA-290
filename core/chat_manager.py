from typing import List, Dict, Optional
from datetime import datetime
import json
import os

class ChatManager:
    """
    Manages conversation history with context-aware retrieval
    Supports multi-turn conversations with memory
    """
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.conversations: Dict[str, List[Dict]] = {}
        
    def add_message(self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to conversation history"""
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.conversations[session_id].append(message)
        
        # Trim history if too long
        if len(self.conversations[session_id]) > self.max_history * 2:
            system_msgs = [m for m in self.conversations[session_id] if m["role"] == "system"]
            recent_msgs = [m for m in self.conversations[session_id] if m["role"] != "system"][-self.max_history * 2:]
            self.conversations[session_id] = system_msgs + recent_msgs
    
    def get_history(self, session_id: str, format_for_llm: bool = False) -> List[Dict]:
        """Get conversation history for a session"""
        history = self.conversations.get(session_id, [])
        
        if format_for_llm:
            return [{"role": m["role"], "content": m["content"]} for m in history]
        
        return history
    
    def get_context_summary(self, session_id: str, last_n: int = 3) -> str:
        """Get a summary of recent conversation for context"""
        history = self.conversations.get(session_id, [])
        recent = history[-last_n:] if len(history) > last_n else history
        
        summary_parts = []
        for msg in recent:
            if msg["role"] == "user":
                summary_parts.append(f"User asked: {msg['content'][:100]}")
            elif msg["role"] == "assistant":
                summary_parts.append(f"Assistant answered about: {msg['content'][:100]}")
        
        return "\n".join(summary_parts)
    
    def clear_session(self, session_id: str):
        """Clear conversation history for a session"""
        if session_id in self.conversations:
            del self.conversations[session_id]
    
    def export_session(self, session_id: str, filepath: Optional[str] = None) -> str:
        """Export conversation to JSON file"""
        if filepath is None:
            os.makedirs("exports", exist_ok=True)
            filepath = f"exports/chat_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        history = self.conversations.get(session_id, [])
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        
        return filepath
