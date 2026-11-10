"""
Base de conocimiento del Agente de Investigación: el manual técnico (ficticio) del ERP de
OLA Muebles que se indexó en Pinecone en la pre-entrega 4.

Dos backends con la misma interfaz `buscar(consulta, k) -> list[Fragmento]`:
- PineconeKB: búsqueda vectorial sobre el índice de la pre-entrega 4 (embeddings de OpenAI).
- LocalKB:    BM25 sobre los mismos archivos del manual (sin servicios externos; útil sin keys).
"""

import asyncio
import json
import logging
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Protocol, TypedDict

from app import config
from app.observability import observar

logger = logging.getLogger("conocimiento")


class Fragmento(TypedDict):
    fuente: str        # archivo del manual (ej.: 03-api-ventas.md)
    titulo: str
    texto: str
    score: float


class KnowledgeBase(Protocol):
    async def buscar(self, consulta: str, k: int = config.TOP_K) -> list[Fragmento]: ...


# --- Pinecone (pre-entrega 4) --------------------------------------------------------
class PineconeKB:
    def __init__(self) -> None:
        from langchain_openai import OpenAIEmbeddings
        from langchain_pinecone import PineconeVectorStore
        from pinecone import Pinecone

        index = Pinecone().Index(config.INDEX_NAME)   # lee PINECONE_API_KEY del entorno
        self.store = PineconeVectorStore(index=index, embedding=OpenAIEmbeddings(model=config.EMBEDDING_MODEL),
                                         text_key="text", namespace=config.PINECONE_NAMESPACE)

    @observar("pinecone.similarity_search", kind="RETRIEVER")
    async def buscar(self, consulta: str, k: int = config.TOP_K) -> list[Fragmento]:
        resultados = await asyncio.to_thread(self.store.similarity_search_with_score, consulta, k)
        return [{"fuente": d.metadata.get("source", "?"), "titulo": d.metadata.get("title", ""),
                 "texto": " ".join(d.page_content.split()), "score": round(float(s), 3)} for d, s in resultados]


# --- Local (BM25 sobre los archivos del manual) ----------------------------------------
STOPWORDS = set("""a al algo como con cual cuando de del desde donde el ella en entre es esa ese esta este hay la las
le lo los mas me mi no o para pero por que se si sin sobre su sus un una uno y ya hace hacer tiene tienen son ser
está están puede pueden quien quién cuánto cuanto cuántos cuantos según""".split())


def tokenize(texto: str) -> list[str]:
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return [t for t in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", texto) if t not in STOPWORDS]


def _fragmentos_md(path: Path) -> list[Fragmento]:
    texto = path.read_text(encoding="utf-8")
    titulo = path.stem
    if texto.startswith("---"):
        _, frontmatter, texto = texto.split("---", 2)
        if m := re.search(r"^title:\s*(.+)$", frontmatter, re.M):
            titulo = m.group(1).strip()
    secciones = re.split(r"\n(?=## )", texto)          # un fragmento por sección "## ..."
    return [{"fuente": path.name, "titulo": titulo, "texto": " ".join(s.split()), "score": 0.0}
            for s in secciones if s.strip()]


def _fragmentos_json(path: Path) -> list[Fragmento]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [{"fuente": path.name, "titulo": data.get("title", path.stem),
             "texto": " ".join(f"{k}: {v}" for k, v in e.items()), "score": 0.0} for e in data.get("entries", [])]


class LocalKB:
    def __init__(self, manual_dir: Path = config.MANUAL_DIR) -> None:
        from rank_bm25 import BM25Okapi

        if not manual_dir.exists():
            raise FileNotFoundError(f"No encuentro el manual en {manual_dir} (configurá MANUAL_DIR)")
        self.fragmentos: list[Fragmento] = []
        for path in sorted(manual_dir.iterdir()):
            if path.suffix == ".md":
                self.fragmentos += _fragmentos_md(path)
            elif path.suffix == ".json" and not path.name.startswith("_"):
                self.fragmentos += _fragmentos_json(path)
        self.bm25 = BM25Okapi([tokenize(f["titulo"] + " " + f["texto"]) for f in self.fragmentos])

    @observar("bm25.search", kind="RETRIEVER")
    async def buscar(self, consulta: str, k: int = config.TOP_K) -> list[Fragmento]:
        scores = self.bm25.get_scores(tokenize(consulta))
        ranking = sorted(zip(scores, self.fragmentos), key=lambda x: -x[0])[:k]
        return [{**f, "score": round(float(s), 3)} for s, f in ranking if s > 0]


@lru_cache(maxsize=1)
def get_kb() -> KnowledgeBase:
    kb: KnowledgeBase = PineconeKB() if config.RESEARCH_BACKEND == "pinecone" else LocalKB()
    logger.info("Base de conocimiento: %s", type(kb).__name__)
    return kb
