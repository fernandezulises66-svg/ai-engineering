"""Configuración compartida por la ingesta y la cadena RAG.

Centralizar el modelo de embeddings acá evita el error #1 de un RAG:
indexar con un modelo y consultar con otro.
"""

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", BASE_DIR / "vectorstore"))
COLLECTION_NAME = "ola_muebles"

# Embeddings: el MISMO modelo para indexar y para consultar
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Chunking: tamaño medido en tokens (no en caracteres)
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
TOKEN_ENCODING = "cl100k_base"

# Recuperación: top_k chico para evitar el "contexto infinito" (lost in the middle)
TOP_K = 4

LLM_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
for ruidoso in ("httpx", "chromadb", "openai", "anthropic"):
    logging.getLogger(ruidoso).setLevel(logging.WARNING)


@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def get_llm():
    from langchain_anthropic import ChatAnthropic

    params: dict[str, Any] = {"model": LLM_MODEL, "max_tokens": 1024, "max_retries": 2, "timeout": 60}
    # Claude 5.x rechaza temperature (error 400): solo se envía si se configura a mano
    if temperature := os.getenv("ANTHROPIC_TEMPERATURE"):
        params["temperature"] = float(temperature)
    return ChatAnthropic(**params)
