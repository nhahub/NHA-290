"""Flask API for AI Health Assistant.

Run:
    python ai_health_assistant/app.py

POST /chat
Body: {"query": "..."}
"""

from __future__ import annotations

import json
from typing import Any, Dict

from flask import Flask, jsonify, request
from flask_cors import CORS

from .agent import get_agent_executor, execute_with_routing


app = Flask(__name__)
CORS(app)

# Lazy-create the agent on first request to allow environment to load
_AGENT = None


def _get_agent():
    global _AGENT
    if _AGENT is None:
        _AGENT = get_agent_executor()
    return _AGENT


@app.route("/health", methods=["GET"])  # Simple health check
def health() -> Any:
    return jsonify({"status": "ok"})


@app.route("/chat", methods=["POST"])
def chat() -> Any:
    try:
        payload: Dict[str, Any] = request.get_json(force=True) or {}
        query: str = str(payload.get("query", "")).strip()
        if not query:
            return jsonify({"error": "Missing 'query'"}), 400

        agent = _get_agent()
        result = execute_with_routing(agent, query)

        # result may be {"output": {"type":..., "content":...}} from our routed wrapper
        if isinstance(result, dict) and "output" in result and isinstance(result["output"], dict):
            return jsonify({"response": result["output"]["content"], "meta": result["output"]["type"]})

        # Fallback: standard agent output
        text = result.get("output") if isinstance(result, dict) else str(result)
        return jsonify({"response": text})
    except Exception as exc:  # Production: log traceback to a real logger
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    # Allow running directly: python ai_health_assistant/app.py
    app.run(host="0.0.0.0", port=8000, debug=False)


