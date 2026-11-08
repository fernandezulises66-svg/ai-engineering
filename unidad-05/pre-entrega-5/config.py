"""Configuración del agente: modelo, rutas y límites."""

import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv
from langchain_core.language_models import BaseChatModel

load_dotenv(find_dotenv())

# La consola de Windows (cp1252) no imprime algunos símbolos: forzamos UTF-8
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("ERP_DB_PATH", BASE_DIR / "erp_demo.db"))              # "base de datos" del ERP
CHECKPOINT_DB = Path(os.getenv("CHECKPOINT_DB", BASE_DIR / "checkpoints.sqlite"))  # memoria del agente
TRAZAS_DIR = BASE_DIR / "trazas"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()
RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", "10"))   # techo de pasos del grafo por turno
MAX_CONTEXT_MESSAGES = 20                                    # mensajes que se envían al modelo en cada paso

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                    datefmt="%H:%M:%S")
for ruidoso in ("httpx", "httpx2", "anthropic", "openai", "aiosqlite"):
    logging.getLogger(ruidoso).setLevel(logging.WARNING)


def get_llm() -> BaseChatModel:
    """Devuelve el chat model configurado (Anthropic por defecto, u OpenAI)."""
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-6-luna"), max_retries=2, timeout=60)

    from langchain_anthropic import ChatAnthropic

    params: dict[str, Any] = {"model": os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5"),
                              "max_tokens": 1024, "max_retries": 2, "timeout": 60}
    if workspace := os.getenv("ANTHROPIC_WORKSPACE_ID"):   # keys no asociadas a un workspace
        params["default_headers"] = {"anthropic-workspace-id": workspace}
    return ChatAnthropic(**params)
