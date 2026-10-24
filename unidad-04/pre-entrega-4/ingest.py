"""
Pipeline de ingesta: lee Markdown, JSON y PDF de /data, los divide en chunks y
los sube a Pinecone con metadatos avanzados.

    python ingest.py           # upsert (idempotente: mismos ids = se sobrescriben)
    python ingest.py --reset   # borra el namespace y vuelve a subir todo

Cada vector guarda en su metadata el texto original del chunk (`text`), la fuente,
la página, la categoría y las etiquetas: así una consulta devuelve todo lo necesario
sin ir a buscar el texto a otra base de datos.
"""

import argparse
import json
import logging
import re
import time
from collections.abc import Callable
from pathlib import Path

import yaml
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

logger = logging.getLogger("ingesta")

SEPARADORES = ["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]


# --- 1. Carga por formato ---------------------------------------------------------
def _base_metadata(meta: dict, source: str) -> dict:
    return {
        "doc_id": meta["doc_id"],
        "title": meta.get("title", source),
        "source": source,
        "category": meta.get("category", "general"),
        "tags": [str(t) for t in meta.get("tags", [])],
    }


def load_markdown(path: Path) -> list[Document]:
    texto = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n", texto, re.DOTALL)
    if not match:
        raise ValueError(f"{path.name}: falta el bloque de metadatos (front matter) con doc_id")
    meta = yaml.safe_load(match.group(1))
    cuerpo = texto[match.end():].strip()
    return [Document(page_content=cuerpo, metadata={**_base_metadata(meta, path.name), "page": 1})]


def load_json(path: Path) -> list[Document]:
    """Un documento por módulo, con todas sus entradas: chunks con contexto suficiente."""
    data = json.loads(path.read_text(encoding="utf-8"))
    por_modulo: dict[str, list[str]] = {}
    for e in data["entries"]:
        por_modulo.setdefault(e["modulo"], []).append(
            f"Código {e['codigo']} ({e['modulo']}): {e['descripcion']} Solución: {e['solucion']}"
        )
    docs = []
    for i, (modulo, lineas) in enumerate(por_modulo.items(), start=1):
        texto = f"# {data['title']} — {modulo}\n\n" + "\n".join(lineas)
        meta = _base_metadata(data, path.name) | {"page": i}
        meta["tags"] = meta["tags"] + [modulo.lower()]
        docs.append(Document(page_content=texto, metadata=meta))
    return docs


def load_pdf(path: Path, sidecar: dict) -> list[Document]:
    from pypdf import PdfReader

    if path.name not in sidecar:
        raise ValueError(f"{path.name}: agregá sus metadatos (doc_id, category...) en data/_metadata.json")
    docs = []
    for numero, page in enumerate(PdfReader(str(path)).pages, start=1):
        texto = page.extract_text() or ""
        texto = re.sub(r"-\n(?=[A-Z0-9])", "-", texto)  # repara códigos cortados: "ERR-\nAFIP-102"
        if texto.strip():
            docs.append(Document(page_content=texto.strip(),
                                 metadata={**_base_metadata(sidecar[path.name], path.name), "page": numero}))
    return docs


def load_documents(data_dir: Path = config.DATA_DIR) -> list[Document]:
    sidecar_path = data_dir / "_metadata.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8")) if sidecar_path.exists() else {}
    docs: list[Document] = []
    for path in sorted(data_dir.iterdir()):
        if path.name.startswith("_"):
            continue
        if path.suffix == ".md":
            docs += load_markdown(path)
        elif path.suffix == ".json":
            docs += load_json(path)
        elif path.suffix == ".pdf":
            docs += load_pdf(path, sidecar)
    logger.info("Cargadas %d unidades (páginas/secciones) de %d archivos",
                len(docs), len({d.metadata["source"] for d in docs}))
    return docs


# --- 2. Chunking ---------------------------------------------------------------------
def approx_tokens(texto: str) -> int:
    return max(1, len(texto) // 4)


def build_splitter(token_counter: Callable[[str], int] | None = None) -> RecursiveCharacterTextSplitter:
    if token_counter is None:
        try:
            return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name=config.TOKEN_ENCODING, chunk_size=config.CHUNK_SIZE_TOKENS,
                chunk_overlap=config.CHUNK_OVERLAP_TOKENS, separators=SEPARADORES)
        except Exception as exc:  # tiktoken descarga su vocabulario la primera vez; sin red, aproximamos
            logger.warning("No se pudo cargar tiktoken (%s): se aproxima 1 token ≈ 4 caracteres", type(exc).__name__)
            token_counter = approx_tokens
    return RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE_TOKENS, chunk_overlap=config.CHUNK_OVERLAP_TOKENS,
                                          length_function=token_counter, separators=SEPARADORES)


def split_documents(docs: list[Document], splitter: RecursiveCharacterTextSplitter | None = None) -> list[Document]:
    chunks = (splitter or build_splitter()).split_documents(docs)
    contador: dict[tuple, int] = {}
    for c in chunks:
        clave = (c.metadata["doc_id"], c.metadata["page"])
        contador[clave] = contador.get(clave, 0) + 1
        c.metadata["chunk"] = contador[clave]
        # id estable: volver a ingerir sobrescribe en lugar de duplicar
        c.metadata["chunk_id"] = f"{c.metadata['doc_id']}#p{c.metadata['page']}-c{contador[clave]}"
    logger.info("Generados %d chunks (%d tokens, overlap %d)", len(chunks), config.CHUNK_SIZE_TOKENS,
                config.CHUNK_OVERLAP_TOKENS)
    return chunks


def load_chunks(splitter: RecursiveCharacterTextSplitter | None = None) -> list[Document]:
    """Mismo proceso para la ingesta y para el índice BM25 local: los chunk_id coinciden."""
    return split_documents(load_documents(), splitter)


# --- 3. Subida a Pinecone ------------------------------------------------------------
def ingest(reset: bool = False) -> int:
    from langchain_pinecone import PineconeVectorStore

    from setup_index import ensure_index

    index = ensure_index()
    if reset and config.NAMESPACE in (index.describe_index_stats().namespaces or {}):
        logger.info("Borrando el namespace '%s'", config.NAMESPACE)
        index.delete(delete_all=True, namespace=config.NAMESPACE)

    chunks = load_chunks()
    store = PineconeVectorStore(index=index, embedding=config.get_embeddings(), text_key="text",
                                namespace=config.NAMESPACE)
    store.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks], batch_size=64)

    # Pinecone es eventualmente consistente: esperamos a que el conteo se actualice
    for _ in range(15):
        ns = (index.describe_index_stats().namespaces or {}).get(config.NAMESPACE)
        if ns and ns.vector_count >= len(chunks):
            break
        time.sleep(2)
    total = ns.vector_count if ns else 0
    logger.info("Subidos %d chunks al índice '%s', namespace '%s' (vectores en el namespace: %d)",
                len(chunks), config.INDEX_NAME, config.NAMESPACE, total)
    return len(chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingesta de documentos en Pinecone")
    parser.add_argument("--reset", action="store_true", help="borrar el namespace antes de subir")
    ingest(reset=parser.parse_args().reset)
