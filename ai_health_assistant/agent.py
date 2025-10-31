"""Orchestrator agent with Gemini, memory, tools, and routing logic.

Provides a callable `get_agent_executor()` that returns a configured
LangChain AgentExecutor ready to handle clinical decision support queries.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict

from langchain.agents import AgentExecutor, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from .config import SETTINGS
from .tools import ALL_TOOLS, symptom_to_care_tool, ddi_checker_tool, medical_rag_tool, web_search_tool


def build_llm() -> BaseChatModel:
    """Create the base LLM (Gemini) with required settings.

    Returns:
        A LangChain chat model instance.
    """
    llm = ChatGoogleGenerativeAI(
        model=SETTINGS.model_name,
        temperature=SETTINGS.temperature,
        google_api_key=SETTINGS.google_api_key,
    )

    # === INSERT YOUR FINE-TUNED MODEL HERE ===
    # Example replacement:
    # llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-finetuned", temperature=0.2)

    return llm


def _rule_based_router(query: str) -> str | None:
    """Lightweight router to direct clear-cut intents to a specific tool.

    Returns one of: "symptom", "ddi", "rag", "web" or None for agent-driven.
    """
    q = query.lower()

    # Drug interaction keywords
    if any(k in q for k in ["interaction", "interact", "safe with", "mix with", "ddi"]):
        return "ddi"
    if any(k in q for k in ["drug", "medication"]) and any(
        k in q for k in ["together", "combine", "with"]
    ):
        return "ddi"

    # Symptom guidance keywords
    if any(k in q for k in ["symptom", "feel", "pain", "fever", "cough", "sore throat", "nausea", "headache"]):
        return "symptom"

    # General medical info likely benefits from RAG
    if any(k in q for k in ["what is", "guideline", "evidence", "treatment", "information", "define"]):
        return "rag"

    # Unknown/novel keywords triggers web
    if any(k in q for k in ["latest", "new", "recent", "outbreak", "recall"]):
        return "web"

    return None


def _wrap_response(content: str, kind: str = "info") -> Dict[str, Any]:
    return {"type": kind, "content": content}


def get_agent_executor() -> AgentExecutor:
    """Create an AgentExecutor with memory and tools.

    The agent uses CONVERSATIONAL_REACT_DESCRIPTION to enable tool usage.
    A light rule-based router handles obvious cases before invoking the agent.
    """
    llm = build_llm()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="input",
        output_key="output",
    )

    agent = initialize_agent(
        tools=ALL_TOOLS,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True,
        memory=memory,
    )

    return agent


def execute_with_routing(agent: AgentExecutor, query: str) -> Dict[str, Any]:
    """Execute a query using lightweight routing, with broad compatibility.

    Falls back to agent.run(query) when .invoke is unavailable.
    Returns a dict aligned with app expectations: {"output": {type, content}} or
    {"output": "..."} depending on backend behavior.
    """
    route = _rule_based_router(query)
    if route == "ddi":
        tool_json = ddi_checker_tool.run(query)
        data = json.loads(tool_json)
        content = (
            f"Risk: {data['risk_level']}\nAdvice: {data['advice']}\n"
            f"Interactions: {json.dumps(data['interactions'], ensure_ascii=False)}"
        )
        return {"output": _wrap_response(content, "warning" if data["risk_level"] != "none" else "info")}
    if route == "symptom":
        tool_json = symptom_to_care_tool.run(query)
        data = json.loads(tool_json)
        lines = [
            "Potential causes: " + ", ".join(data.get("potential_causes", [])),
            "Self-care: " + "; ".join(data.get("self_care", [])),
        ]
        if data.get("red_flags"):
            lines.append("Red flags: " + ", ".join(data["red_flags"]))
        lines.append(f"Disposition: {data['disposition']}")
        return {"output": _wrap_response("\n".join(lines), "advice")}
    if route == "rag":
        tool_json = medical_rag_tool.run(query)
        data = json.loads(tool_json)
        snippets = [m.get("text", "").strip() for m in data.get("matches", [])]
        content = "\n---\n".join(snippets[:3]) or "No context found."
        return {"output": _wrap_response(content, "info")}
    if route == "web":
        tool_json = web_search_tool.run(query)
        data = json.loads(tool_json)
        return {"output": _wrap_response(str(data.get("result", "")), "info")}

    # Agent fallbacks for different LC versions
    if hasattr(agent, "invoke"):
        return agent.invoke({"input": query})  # type: ignore[no-any-return]
    if hasattr(agent, "run"):
        text = agent.run(query)
        return {"output": text}
    # Generic call protocol
    return agent({"input": query})  # type: ignore[no-any-return]


