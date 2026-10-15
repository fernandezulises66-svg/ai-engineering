"""
Tests sin API keys: embeddings determinísticos falsos y un LLM falso que devuelve
JSON predefinido. Verifican la ingesta (chunking + persistencia), la reutilización
de la base y el armado de la respuesta grounded.

    pytest -v
"""

import asyncio
import json

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel

import config
from ingest import build_splitter, ingest, load_documents, split_documents
from rag import NO_LO_SE, build_rag_chain, get_rag_response


def contar_tokens_aprox(texto: str) -> int:
    """Aproximación (1 token ≈ 4 caracteres) para no depender de descargar tiktoken."""
    return max(1, len(texto) // 4)


class EmbeddingsContador(DeterministicFakeEmbedding):
    llamadas: int = 0

    def embed_documents(self, texts):
        self.llamadas += 1
        return super().embed_documents(texts)


@pytest.fixture
def splitter():
    return build_splitter(token_counter=contar_tokens_aprox)


@pytest.fixture
def store(tmp_path, splitter):
    return ingest(persist_dir=tmp_path, embeddings=DeterministicFakeEmbedding(size=64), splitter=splitter)


# --- Ingesta -------------------------------------------------------------------

def test_carga_los_cuatro_documentos():
    docs = load_documents()
    assert [d.metadata["source"] for d in docs] == sorted(p.name for p in config.DATA_DIR.glob("*.md"))
    assert all(d.metadata["title"].startswith("OLA Muebles") for d in docs)


def test_chunks_respetan_tamano_y_tienen_ids_unicos(splitter):
    chunks = split_documents(load_documents(), splitter)
    assert len(chunks) > len(load_documents())  # cada documento se partió en varios chunks
    assert all(contar_tokens_aprox(c.page_content) <= config.CHUNK_SIZE_TOKENS for c in chunks)
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_no_reindexa_si_la_base_esta_al_dia(tmp_path, splitter):
    emb = EmbeddingsContador(size=64)
    primera = ingest(persist_dir=tmp_path, embeddings=emb, splitter=splitter)
    total = primera._collection.count()
    segunda = ingest(persist_dir=tmp_path, embeddings=emb, splitter=splitter)
    assert emb.llamadas == 1  # la segunda vez no se volvió a embeber nada
    assert segunda._collection.count() == total


def test_reindexa_con_force_o_si_cambia_el_modelo(tmp_path, splitter):
    emb = EmbeddingsContador(size=64)
    ingest(persist_dir=tmp_path, embeddings=emb, splitter=splitter)
    ingest(persist_dir=tmp_path, embeddings=emb, splitter=splitter, force=True)
    ingest(persist_dir=tmp_path, embeddings=emb, splitter=splitter, embedding_model="otro-modelo")
    assert emb.llamadas == 3


# --- Cadena RAG ----------------------------------------------------------------

def respuesta_json(respuesta: str, encontrada: bool, fragmentos: list[int]) -> str:
    return json.dumps({"respuesta": respuesta, "encontrada_en_contexto": encontrada, "fragmentos_usados": fragmentos})


def preguntar(store, *respuestas_llm: str, pregunta="¿Qué garantía tienen los herrajes?"):
    llm = FakeListChatModel(responses=list(respuestas_llm))
    return asyncio.run(get_rag_response(pregunta, chain=build_rag_chain(store, llm=llm, k=4)))


def test_respuesta_con_referencias_a_fragmentos_reales(store):
    r = preguntar(store, respuesta_json("Los herrajes tienen 6 meses de garantía.", True, [1, 1, 2, 9]))
    assert r.encontrada_en_contexto
    assert r.respuesta == "Los herrajes tienen 6 meses de garantía."
    assert len(r.referencias) == 2  # se descarta el duplicado y el [9], que no existe
    assert all(ref.fuente.endswith(".md") and ref.score is not None for ref in r.referencias)


def test_pregunta_trampa_devuelve_no_lo_se_sin_referencias(store):
    r = preguntar(store, respuesta_json("Creo que 12 cuotas.", False, [1]), pregunta="¿Cuántas cuotas sin interés hay?")
    assert not r.encontrada_en_contexto
    assert r.respuesta == NO_LO_SE
    assert r.referencias == []


def test_reintenta_si_el_modelo_devuelve_json_invalido(store):
    r = preguntar(store, "esto no es JSON", respuesta_json("6 meses.", True, [1]))
    assert r.encontrada_en_contexto and r.respuesta == "6 meses."


def test_pregunta_vacia(store):
    with pytest.raises(ValueError):
        preguntar(store, respuesta_json("x", True, [1]), pregunta="  ")
