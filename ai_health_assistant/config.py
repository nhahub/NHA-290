"""Configuration and environment loading for AI Health Assistant.

This module centralizes environment variable access and configuration
for the application. It uses python-dotenv to load variables from a
.env file if present.

Required/optional environment variables:
- GOOGLE_API_KEY (required for Gemini + embeddings)
- SERPAPI_API_KEY (optional; enables real web search)
- OPENAI_API_KEY (optional; supports alternative embeddings if desired)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


load_dotenv()  # Load .env if present


@dataclass(frozen=True)
class Settings:
    """Strongly-typed application settings."""

    google_api_key: str
    serpapi_api_key: Optional[str]
    openai_api_key: Optional[str]
    model_name: str
    temperature: float


def get_settings() -> Settings:
    """Read environment variables and return an immutable Settings object.

    Raises:
        RuntimeError: If required environment variables are missing.
    """

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is required. Set it in your environment or .env file."
        )

    serpapi_api_key = os.getenv("SERPAPI_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Default Gemini model and temperature as specified
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
    temperature_str = os.getenv("MODEL_TEMPERATURE", "0.2")
    try:
        temperature = float(temperature_str)
    except ValueError as exc:
        raise RuntimeError("MODEL_TEMPERATURE must be a float") from exc

    return Settings(
        google_api_key=google_api_key,
        serpapi_api_key=serpapi_api_key,
        openai_api_key=openai_api_key,
        model_name=model_name,
        temperature=temperature,
    )


# Global settings instance for convenience
SETTINGS = get_settings()


