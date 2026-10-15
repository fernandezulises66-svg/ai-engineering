"""
Módulo de ingesta: lee los .md/.txt de /data, los fragmenta por tokens y los
persiste en una colección local de ChromaDB.

    python ingest.py           # indexa solo si hace falta
    python ingest.py --force   # reindexa todo desde cero

Antes de reindexar, compara una "huella" (hash) de los documentos, de los
parámetros de chunking y del modelo de embeddings con la guardada en la
colección. Si nada cambió, reutiliza la base existente (ahorra tiempo y costo).
"""

import argparse
import hashlib
import logging
from collections.abc import Callable
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

logger = logging.getLogger("ingesta")

SEPARADORES_MARKDOWN = ["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]


# --- 1. Carga -----------------------------------------------------------------
def load_documents(data_dir: Path = config.DATA_DIR) -> list[Document]:
    archivos = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in {".md", ".txt"})
    if not archivos:
        raise FileNotFoundError(f"No hay archivos .md o .txt en {data_dir}")

    docs = []
    for path in archivos:
        texto = path.read_text(encoding="utf-8")
        titulo = next((l.lstrip("# ").strip() for l in texto.splitlines() if l.startswith("# ")), path.stem)
        docs.append(Document(page_content=texto, metadata={"source": path.name, "title": titulo}))
    logger.info("Cargados %d documentos desde %s", len(docs), data_dir)
    return docs


# --- 2. Chunking por tokens ---------------------------------------------------
def build_splitter(token_counter: Callable[[str], int] | None = None) -> RecursiveCharacterTextSplitter:
    """Splitter de 500 tokens con 50 de overlap, respetando la estructura del Markdown.

    Por defecto cuenta tokens con tiktoken (cl100k_base). `token_counter` permite
    inyectar otro contador (por ejemplo, en los tests offline).
    """
    if token_counter is None:
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=config.TOKEN_ENCODING,
            chunk_size=config.CHUNK_SIZE_TOKENS,
            chunk_overlap=config.CHUNK_OVERLAP_TOKENS,
            separators=SEPARADORES_MARKDOWN,
        )
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE_TOKENS,
        chunk_overlap=config.CHUNK_OVERLAP_TOKENS,
        length_function=token_counter,
        separators=SEPARADORES_MARKDOWN,
    )


def split_documents(docs: list[Document], splitter: RecursiveCharacterTextSplitter | None = None) -> list[Document]:
    splitter = splitter or build_splitter()
    chunks = splitter.split_documents(docs)

    contador: dict[str, int] = {}
    for chunk in chunks:
        fuente = chunk.metadata["source"]
        contador[fuente] = contador.get(fuente, 0) + 1
        chunk.metadata["chunk"] = contador[fuente]
        chunk.metadata["chunk_id"] = f"{fuente}#{contador[fuente]}"

    logger.info("Generados %d chunks (%d tokens, overlap %d)",
                len(chunks), config.CHUNK_SIZE_TOKENS, config.CHUNK_OVERLAP_TOKENS)
    return chunks


# --- 3. Persistencia en ChromaDB ----------------------------------------------
def fingerprint(docs: list[Document], embedding_model: str) -> str:
    h = hashlib.sha256()
    h.update(f"{embedding_model}|{config.CHUNK_SIZE_TOKENS}|{config.CHUNK_OVERLAP_TOKENS}".encode())
    for doc in docs:
        h.update(doc.metadata["source"].encode())
        h.update(doc.page_content.encode())
    return h.hexdigest()[:16]


def _client(persist_dir: Path) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(persist_dir), settings=Settings(anonymized_telemetry=False))


def _open_store(client, embeddings: Embeddings, metadata: dict | None = None) -> Chroma:
    return Chroma(
        client=client,
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        collection_metadata=metadata,
        collection_configuration={"hnsw": {"space": "cosine"}},  # similitud coseno
    )


def ingest(
    force: bool = False,
    data_dir: Path = config.DATA_DIR,
    persist_dir: Path = config.VECTORSTORE_DIR,
    embeddings: Embeddings | None = None,
    embedding_model: str = config.EMBEDDING_MODEL,
    splitter: RecursiveCharacterTextSplitter | None = None,
) -> Chroma:
    """Indexa los documentos si hace falta y devuelve el vectorstore listo para consultar."""
    embeddings = embeddings or config.get_embeddings()
    docs = load_documents(data_dir)
    huella = fingerprint(docs, embedding_model)
    client = _client(persist_dir)

    existentes = {c.name for c in client.list_collections()}
    if config.COLLECTION_NAME in existentes:
        coleccion = client.get_collection(config.COLLECTION_NAME)
        meta = coleccion.metadata or {}
        if not force and meta.get("fingerprint") == huella and coleccion.count() > 0:
            logger.info("La base ya existe y está al día (%d chunks, huella %s): se reutiliza sin reindexar",
                        coleccion.count(), huella)
            return _open_store(client, embeddings)

        if meta.get("embedding_model") not in (None, embedding_model):
            logger.warning("La base fue indexada con '%s' y ahora se usa '%s': se reindexa para no mezclar espacios vectoriales",
                           meta.get("embedding_model"), embedding_model)
        else:
            logger.info("Cambiaron los documentos o se pidió --force: se reindexa")
        client.delete_collection(config.COLLECTION_NAME)

    chunks = split_documents(docs, splitter)
    store = _open_store(client, embeddings, metadata={"fingerprint": huella, "embedding_model": embedding_model})
    store.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
    logger.info("Indexados %d chunks en %s (colección '%s', modelo %s)",
                len(chunks), persist_dir, config.COLLECTION_NAME, embedding_model)
    return store


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingesta de documentos en ChromaDB")
    parser.add_argument("--force", action="store_true", help="reindexar aunque la base esté al día")
    ingest(force=parser.parse_args().force)
