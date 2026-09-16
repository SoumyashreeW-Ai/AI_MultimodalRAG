"""Configuration foundation for the application.

This module uses Pydantic when available and falls back to a lightweight
implementation so that import tests do not fail in minimal environments.
"""
from __future__ import annotations

from pathlib import Path
import os
from typing import Optional

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - dotenv optional in some envs
    load_dotenv = None

try:
    from pydantic import BaseSettings

    class Settings(BaseSettings):
        app_name: str = "multimodal-rag"
        environment: str = "development"
        openai_api_key: str | None = None
        chroma_db_path: Path = Path("data/chroma")
        upload_dir: Path = Path("data/uploads")
        text_collection_name: str = "text_chunks"
        text_embedding_model: str = "all-MiniLM-L6-v2"
        image_embedding_model: str = "clip"
        multimodal_embedding_model: str = "clip"
        chunk_size: int = 500
        chunk_overlap: int = 50
        text_weight: float = 1.0
        image_weight: float = 1.0
        # LLM / generation settings
        llm_provider: str | None = None
        llm_api_key: str | None = None
        llm_model: str | None = None
        llm_base_url: str | None = None
        llm_temperature: float = 0.0
        llm_max_tokens: int = 1024
        vision_enabled: bool = False

        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"

except Exception:  # pragma: no cover - fallback for environments without pydantic
    from dataclasses import dataclass

    @dataclass
    class Settings:
        app_name: str = "multimodal-rag"
        environment: str = "development"
        openai_api_key: str | None = None
        chroma_db_path: Path = Path("data/chroma")
        upload_dir: Path = Path("data/uploads")
        text_collection_name: str = "text_chunks"
        text_embedding_model: str = "all-MiniLM-L6-v2"
        image_embedding_model: str = "clip"
        multimodal_embedding_model: str = "clip"
        chunk_size: int = 500
        chunk_overlap: int = 50
        text_weight: float = 1.0
        image_weight: float = 1.0
        # LLM / generation settings (dataclass fallback)
        llm_provider: str | None = None
        llm_api_key: str | None = None
        llm_model: str | None = None
        llm_base_url: str | None = None
        llm_temperature: float = 0.0
        llm_max_tokens: int = 1024
        vision_enabled: bool = False


# Simple accessor
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        # Ensure .env from project root (or parents) is loaded into the environment.
        # Strategy: look for a `.env` file starting from cwd and from this file's project root.
        dotenv_path: Optional[Path] = None
        # 1) search from current working directory upwards
        try:
            cwd = Path(os.getcwd()).resolve()
            for p in [cwd] + list(cwd.parents):
                candidate = p / ".env"
                if candidate.exists():
                    dotenv_path = candidate
                    break
        except Exception:
            dotenv_path = None

        # 2) if not found, search from the package project root (two parents up)
        if dotenv_path is None:
            try:
                pkg_root = Path(__file__).resolve().parents[2]
                for p in [pkg_root] + list(pkg_root.parents):
                    candidate = p / ".env"
                    if candidate.exists():
                        dotenv_path = candidate
                        break
            except Exception:
                dotenv_path = None

        if dotenv_path and load_dotenv is not None:
            # load into the environment without overwriting existing vars
            load_dotenv(dotenv_path, override=False)

        try:
            _settings = Settings()
        except Exception:
            # In constrained environments, construct dataclass fallback manually
            _settings = Settings()
        # Backfill settings from environment variables to be robust across pydantic versions
        try:
            # map env var names to settings attributes
            env_map = {
                "LLM_PROVIDER": ("llm_provider", str),
                "LLM_API_KEY": ("llm_api_key", str),
                "LLM_MODEL": ("llm_model", str),
                "LLM_BASE_URL": ("llm_base_url", str),
                "LLM_TEMPERATURE": ("llm_temperature", float),
                "LLM_MAX_TOKENS": ("llm_max_tokens", int),
                "VISION_ENABLED": ("vision_enabled", lambda v: v.lower() in ("1", "true", "yes", "on")),
                "TEXT_EMBEDDING_MODEL": ("text_embedding_model", str),
                "TEXT_COLLECTION_NAME": ("text_collection_name", str),
            }
            for env_name, (attr, cast) in env_map.items():
                val = os.environ.get(env_name)
                if val is not None and (
                    env_name in {"TEXT_EMBEDDING_MODEL", "TEXT_COLLECTION_NAME"}
                    or getattr(_settings, attr, None) in (None, "", 0)
                ):
                    try:
                        setattr(_settings, attr, cast(val))
                    except Exception:
                        # last-resort: set raw string for non-castable values
                        setattr(_settings, attr, val)
        except Exception:
            pass
    return _settings


def get_llm_config_status() -> dict:
    """Return safe booleans about LLM configuration without exposing secrets."""
    s = get_settings()
    return {
        "llm_provider_configured": bool(s.llm_provider),
        "llm_model_configured": bool(s.llm_model),
        "llm_api_key_configured": bool(s.llm_api_key),
        "vision_enabled": bool(s.vision_enabled),
    }


def is_rag_debug_enabled() -> bool:
    """Return whether development-only retrieval diagnostics are enabled."""
    # Load the project's .env file before reading the process environment.  Reading
    # this flag directly also means DEBUG_RAG is evaluated for each request.
    get_settings()
    return os.environ.get("DEBUG_RAG", "").strip().lower() == "true"
