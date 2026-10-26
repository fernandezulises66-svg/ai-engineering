"""
RAGSystem: recuperador híbrido que combina BM25 (léxico) con Pinecone (vectorial)
mediante un EnsembleRetriever (fusión por Reciprocal Rank Fusion).

    python rag_system.py "¿Qué significa el error ERR-AFIP-102?"
"""

import logging
import re
import sys
import unicodedata
from typing import Literal

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

import config

logger = logging.getLogger("rag")

Mode = Literal["hybrid_exact", "hybrid", "bm25", "vector"]

# Identificadores: letras/números unidos por guiones y con al menos un dígito (ERR-AFIP-102, OLA-RIB-0042)
IDENTIFIER_RE = re.compile(r"\b(?=[A-Za-z0-9-]*\d)[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b")

STOPWORDS = set("""a al algo como con cual cuando de del desde donde el ella en entre es esa ese esta este hay la las
le lo los mas me mi no o para pero por que se si sin sobre su sus un una uno y ya qué cómo cuál cuándo dónde
hace hacer tiene tienen son ser está están se le""".split())


def tokenize(texto: str) -> list[str]:
    """Tokenizador para BM25 en español técnico.

    - minúsculas y sin acentos ("autenticación" = "autenticacion")
    - conserva códigos con guiones como un token ("err-afip-102") y además
      agrega sus partes ("err", "afip", "102"), para que también matchee "error 102 de AFIP"
    - descarta palabras vacías
    """
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    tokens = []
    for tok in re.findall(r"[a-z0-9]+(?:[-_/][a-z0-9]+)*", texto):
        if tok in STOPWORDS:
            continue
        tokens.append(tok)
        if re.search(r"[-_/]", tok):
            tokens += [p for p in re.split(r"[-_/]", tok) if p and p not in STOPWORDS]
    return tokens


class RAGSystem:
    """Encapsula el EnsembleRetriever: recibe una consulta y devuelve los top-k chunks."""

    def __init__(self, chunks: list[Document], vector_store: VectorStore, k: int = config.TOP_K,
                 candidates: int = config.CANDIDATES_PER_RETRIEVER, weights=config.HYBRID_WEIGHTS):
        self.k = k
        self.chunks = chunks
        self.bm25 = BM25Retriever.from_documents(chunks, k=candidates, preprocess_func=tokenize)
        self.vector = vector_store.as_retriever(search_kwargs={"k": candidates})
        self.ensemble = EnsembleRetriever(retrievers=[self.bm25, self.vector], weights=list(weights),
                                          id_key="chunk_id")  # deduplica por id de chunk

    @classmethod
    def from_pinecone(cls, **kwargs) -> "RAGSystem":
        from langchain_pinecone import PineconeVectorStore

        from ingest import load_chunks

        chunks = load_chunks()  # mismo chunking que la ingesta → mismos chunk_id que en Pinecone
        index = config.get_pinecone().Index(config.INDEX_NAME)
        store = PineconeVectorStore(index=index, embedding=config.get_embeddings(), text_key="text",
                                    namespace=config.NAMESPACE)
        return cls(chunks, store, **kwargs)

    def exact_matches(self, query: str) -> list[Document]:
        """Chunks que contienen literalmente algún identificador de la consulta, ordenados por BM25.

        RRF premia el consenso entre recuperadores: un chunk que BM25 pone 1.º pero que
        la búsqueda vectorial no trae (típico con códigos y SKUs) puede perder contra
        chunks mediocres que aparecen en las dos listas. Si el usuario escribió un
        identificador exacto, los chunks que lo contienen tienen prioridad.
        """
        identificadores = {i.lower() for i in IDENTIFIER_RE.findall(query)}
        if not identificadores:
            return []
        scores = self.bm25.vectorizer.get_scores(tokenize(query))
        candidatos = [(score, c) for score, c in zip(scores, self.chunks)
                      if any(i in c.page_content.lower() for i in identificadores)]
        return [c for _, c in sorted(candidatos, key=lambda x: -x[0])]

    def retrieve(self, query: str, mode: Mode = "hybrid_exact") -> list[Document]:
        if not query.strip():
            raise ValueError("La consulta está vacía")
        if mode == "hybrid_exact":
            exactos = self.exact_matches(query)
            ids = {d.metadata["chunk_id"] for d in exactos}
            docs = (exactos + [d for d in self.ensemble.invoke(query) if d.metadata["chunk_id"] not in ids])[: self.k]
        else:
            retriever = {"hybrid": self.ensemble, "bm25": self.bm25, "vector": self.vector}[mode]
            docs = retriever.invoke(query)[: self.k]
        logger.info("[%s] %s → %s", mode, query[:60], [d.metadata["chunk_id"] for d in docs])
        return docs

    async def aretrieve(self, query: str, mode: Mode = "hybrid_exact") -> list[Document]:
        import asyncio

        return await asyncio.to_thread(self.retrieve, query, mode)  # BM25 es CPU: no bloquea el loop


if __name__ == "__main__":
    consulta = " ".join(sys.argv[1:]) or "¿Qué significa el error ERR-AFIP-102 y cómo se soluciona?"
    rag = RAGSystem.from_pinecone()
    for i, d in enumerate(rag.retrieve(consulta), start=1):
        m = d.metadata
        print(f"{i}. {m['chunk_id']}  [{m['category']}] {m['source']} (pág. {m['page']})")
        print("   " + " ".join(d.page_content.split())[:160] + "…")
