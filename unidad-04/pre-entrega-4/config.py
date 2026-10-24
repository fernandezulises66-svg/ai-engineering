"""Configuración compartida: Pinecone, embeddings, chunking y recuperación."""

import logging
import os
import sys
from functools import lru_cache
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

# La consola de Windows (cp1252) no imprime algunos símbolos: forzamos UTF-8
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# --- Pinecone ------------------------------------------------------------------
INDEX_NAME = os.getenv("INDEX_NAME", "ola-erp-docs")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "erp-docs-v1")  # separa este dataset de otros en el mismo índice
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
METRIC = "cosine"

# --- Embeddings: la dimensión del índice DEBE coincidir con la del modelo ----------
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}
EMBEDDING_DIM = EMBEDDING_DIMENSIONS.get(EMBEDDING_MODEL, int(os.getenv("EMBEDDING_DIM", "1536")))

# --- Chunking (en tokens): punto medio entre contexto y precisión -----------------
CHUNK_SIZE_TOKENS = 600
CHUNK_OVERLAP_TOKENS = 80
TOKEN_ENCODING = "cl100k_base"

# --- Recuperación -----------------------------------------------------------------
TOP_K = 5                 # documentos que devuelve el sistema
CANDIDATES_PER_RETRIEVER = 10  # cuántos aporta cada recuperador antes de fusionar
HYBRID_WEIGHTS = (0.5, 0.5)    # (BM25, vectorial) en la fusión RRF

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                    datefmt="%H:%M:%S")
for ruidoso in ("httpx", "openai", "urllib3", "pinecone"):
    logging.getLogger(ruidoso).setLevel(logging.WARNING)


@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_pinecone():
    from pinecone import Pinecone

    if not os.getenv("PINECONE_API_KEY"):
        raise RuntimeError("Falta PINECONE_API_KEY en el .env")
    return Pinecone()  # lee PINECONE_API_KEY del entorno
