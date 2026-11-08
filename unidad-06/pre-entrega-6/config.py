"""Configuración del orquestador: modelo, base de conocimiento y límites."""

import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv
from langchain_core.language_models import BaseChatModel

load_dotenv(find_dotenv(usecwd=True))

# La consola de Windows (cp1252) no imprime algunos símbolos: forzamos UTF-8
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
TRAZAS_DIR = BASE_DIR / "trazas"

# --- LLM ----------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()

# --- Base de conocimiento del Investigador --------------------------------------------
# "pinecone": el índice de la pre-entrega 4 (manual técnico del ERP de OLA Muebles).
# "local": búsqueda BM25 sobre los mismos archivos del manual, sin servicios externos.
RESEARCH_BACKEND = os.getenv("RESEARCH_BACKEND", "pinecone").lower()
INDEX_NAME = os.getenv("INDEX_NAME", "ola-erp-docs")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "erp-docs-v1")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
MANUAL_DIR = Path(os.getenv("MANUAL_DIR", BASE_DIR.parent.parent / "unidad-04" / "pre-entrega-4" / "data"))
TOP_K = 4

# --- Límites del flujo (evitan el "supervisor infinito") ------------------------------
MAX_DELEGACIONES = int(os.getenv("MAX_DELEGACIONES", "6"))   # veces que el supervisor puede delegar en total
MAX_REVISIONES = 2               # veces que puede devolver el trabajo a un mismo especialista para refinar
RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", "40"))    # techo duro de pasos del grafo
AGENT_RECURSION_LIMIT = 12       # techo de pasos del ciclo ReAct interno de cada especialista

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                    datefmt="%H:%M:%S")
for ruidoso in ("httpx", "httpx2", "anthropic", "openai", "pinecone", "urllib3"):
    logging.getLogger(ruidoso).setLevel(logging.WARNING)


def get_llm(max_tokens: int = 1024) -> BaseChatModel:
    """Devuelve el chat model configurado (Anthropic por defecto, u OpenAI)."""
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-6-luna"), max_retries=2, timeout=60)

    from langchain_anthropic import ChatAnthropic

    params: dict[str, Any] = {"model": os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5"),
                              "max_tokens": max_tokens, "max_retries": 2, "timeout": 60}
    if workspace := os.getenv("ANTHROPIC_WORKSPACE_ID"):   # keys no asociadas a un workspace
        params["default_headers"] = {"anthropic-workspace-id": workspace}
    return ChatAnthropic(**params)
