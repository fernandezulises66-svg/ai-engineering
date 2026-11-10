"""Configuración de la API: modelo, Redis, observabilidad, base de conocimiento y límites."""

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

APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent

# --- LLM ------------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()

# --- Redis: estado de los jobs, cola de trabajo y checkpoints de LangGraph ---------------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
JOB_TTL_SEGUNDOS = int(os.getenv("JOB_TTL_SEGUNDOS", str(7 * 24 * 3600)))   # los jobs viven 7 días
WORKER_CONCURRENCY = int(os.getenv("WORKER_CONCURRENCY", "5"))             # jobs en paralelo por proceso
WORKER_EMBEDDED = os.getenv("WORKER_EMBEDDED", "true").lower() == "true"   # worker dentro del proceso de la API

# --- Observabilidad (Arize Phoenix vía OpenTelemetry / OpenInference) --------------------
PHOENIX_ENABLED = os.getenv("PHOENIX_ENABLED", "true").lower() == "true"
PHOENIX_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces")
PHOENIX_PROJECT = os.getenv("PHOENIX_PROJECT_NAME", "ola-erp-orquestador")

# --- Base de conocimiento del Investigador (índice de la pre-entrega 4) -------------------
RESEARCH_BACKEND = os.getenv("RESEARCH_BACKEND", "pinecone").lower()
INDEX_NAME = os.getenv("INDEX_NAME", "ola-erp-docs")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "erp-docs-v1")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
MANUAL_DIR = Path(os.getenv("MANUAL_DIR", BASE_DIR.parent.parent / "unidad-04" / "pre-entrega-4" / "data"))
TOP_K = 3                         # fragmentos por búsqueda (cada chunk de la PE4 tiene ~2.000 caracteres)
FRAGMENTO_MAX_CHARS = 3000        # no recortar de más: una regla puede estar al final del chunk

# --- Límites del flujo (evitan el "supervisor infinito") -------------------------------
MAX_DELEGACIONES = int(os.getenv("MAX_DELEGACIONES", "6"))
MAX_REVISIONES = 2
RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", "40"))
AGENT_RECURSION_LIMIT = 12

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                    datefmt="%H:%M:%S")
for ruidoso in ("httpx", "httpx2", "anthropic", "openai", "pinecone", "urllib3", "redisvl", "opentelemetry"):
    logging.getLogger(ruidoso).setLevel(logging.WARNING)


def get_llm(max_tokens: int = 2048) -> BaseChatModel:
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
