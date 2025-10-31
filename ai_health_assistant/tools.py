"""Tool layer for the AI Health Assistant.

Implements:
- symptom_to_care_tool: maps symptoms to guidance (simulated)
- ddi_checker_tool: checks for drug-drug interactions (simulated)
- web_search_tool: uses SerpAPI if available, otherwise placeholder
- medical_rag_tool: wraps the FAISS retriever for RAG

All tools return structured, readable responses.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from langchain.tools import tool
from langchain_community.utilities import SerpAPIWrapper

from .config import SETTINGS
from .rag_setup import get_retriever


@tool("symptom_to_care_tool", return_direct=False)
def symptom_to_care_tool(symptoms: str) -> str:
    """Provide symptom-to-care guidance.

    Simulates an Infermedica-style triage. Input is a free-text symptom list.

    Args:
        symptoms: A free-text description of symptoms.

    Returns:
        A JSON string containing: potential_causes, self_care, red_flags, disposition.
    """
    text = symptoms.lower()
    potential_causes: list[str] = []
    red_flags: list[str] = []
    self_care: list[str] = [
        "Rest and stay hydrated",
        "Consider acetaminophen for fever or pain per label dosing",
        "Monitor symptoms over 24-48 hours",
    ]
    disposition = "self_care"

    if any(k in text for k in ["chest pain", "pressure", "tightness"]):
        potential_causes.append("Possible cardiac or musculoskeletal cause")
        red_flags.append("Chest pain may indicate cardiac emergency")
        disposition = "urgent_care"

    if any(k in text for k in ["shortness of breath", "trouble breathing"]):
        potential_causes.append("Respiratory distress or asthma exacerbation")
        red_flags.append("Severe shortness of breath is an emergency")
        disposition = "urgent_care"

    if any(k in text for k in ["fever", "cough", "sore throat", "runny nose"]):
        potential_causes.append("Viral upper respiratory infection (common cold/flu)")

    if "headache" in text:
        potential_causes.append("Tension headache or migraine")

    response: Dict[str, Any] = {
        "tool": "symptom_to_care",
        "input": symptoms,
        "potential_causes": list(dict.fromkeys(potential_causes)),
        "self_care": self_care,
        "red_flags": red_flags,
        "disposition": disposition,
        "disclaimer": (
            "Educational guidance only. Not a diagnosis. Seek medical care for concerns."
        ),
    }
    return json.dumps(response)


@tool("ddi_checker_tool", return_direct=False)
def ddi_checker_tool(drugs: str) -> str:
    """Check drug-drug interactions (simulated).

    Args:
        drugs: A comma-separated list of medication names.

    Returns:
        A JSON string with fields: risk_level, interactions, advice.
    """
    text = drugs.lower()
    listed = [d.strip() for d in drugs.split(",") if d.strip()]

    risk_level = "none"
    interactions: list[dict[str, str]] = []
    advice = "No known significant interactions in simulated dataset. Verify with a pharmacist."

    # Simulated rules
    if "ibuprofen" in text and "aspirin" in text:
        risk_level = "moderate"
        interactions.append(
            {
                "pair": "ibuprofen + aspirin",
                "type": "pharmacodynamic",
                "description": "Increased bleeding risk and reduced antiplatelet effect of aspirin.",
            }
        )
        advice = "Avoid combining regularly; consult clinician about timing and alternatives."

    if "warfarin" in text and ("bactrim" in text or "trimethoprim" in text or "sulfamethoxazole" in text):
        risk_level = "major"
        interactions.append(
            {
                "pair": "warfarin + trimethoprim/sulfamethoxazole",
                "type": "pharmacokinetic",
                "description": "Markedly increases INR; bleeding risk.",
            }
        )
        advice = "Avoid combination; if necessary, close INR monitoring and dose adjustment."

    response = {
        "tool": "ddi_checker",
        "input": listed,
        "risk_level": risk_level,
        "interactions": interactions,
        "advice": advice,
        "disclaimer": (
            "Simulated DDI. Not comprehensive. Consult DrugBank/clinical pharmacist for confirmation."
        ),
    }
    return json.dumps(response)


@tool("web_search_tool", return_direct=False)
def web_search_tool(query: str) -> str:
    """Perform a web search for up-to-date medical information.

    Uses SerpAPI if configured; otherwise returns a placeholder string.

    Args:
        query: Search query string.

    Returns:
        A simple structured JSON string with top result snippets or a placeholder.
    """
    if SETTINGS.serpapi_api_key:
        search = SerpAPIWrapper(serpapi_api_key=SETTINGS.serpapi_api_key)
        result = search.run(query)
        return json.dumps({"tool": "web_search", "query": query, "result": result})
    return json.dumps(
        {
            "tool": "web_search",
            "query": query,
            "result": "Web search not configured. Provide SERPAPI_API_KEY to enable.",
        }
    )


@tool("medical_rag_tool", return_direct=False)
def medical_rag_tool(question: str) -> str:
    """Retrieve trusted, evidence-based health info using FAISS-based RAG.

    Args:
        question: The user's medical information question.

    Returns:
        A JSON string with top contextual passages for grounding.
    """
    retriever = get_retriever()
    docs = retriever.invoke(question)
    payload = [
        {"text": d.page_content, "source": d.metadata.get("source", "unknown")}
        for d in docs
    ]
    return json.dumps({"tool": "medical_rag", "matches": payload})


# Exported list of tools for agent construction
ALL_TOOLS = [
    symptom_to_care_tool,
    ddi_checker_tool,
    web_search_tool,
    medical_rag_tool,
]


