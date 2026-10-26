"""
Tests sin API keys: Pinecone se reemplaza por un vector store en memoria y un
cliente falso, y los embeddings por embeddings determinísticos.

    pytest -v
"""

from types import SimpleNamespace

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.vectorstores import InMemoryVectorStore

import config
from evaluate import evaluate, precision_at_k, recall_at_k, reciprocal_rank
from ingest import approx_tokens, build_splitter, load_documents, split_documents
from rag_system import RAGSystem, tokenize


@pytest.fixture(scope="module")
def chunks():
    return split_documents(load_documents(), build_splitter(token_counter=approx_tokens))


@pytest.fixture(scope="module")
def rag(chunks):
    store = InMemoryVectorStore(DeterministicFakeEmbedding(size=16))
    store.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
    return RAGSystem(chunks, store)


# --- Ingesta ------------------------------------------------------------------------
def test_carga_los_tres_formatos():
    docs = load_documents()
    fuentes = {d.metadata["source"] for d in docs}
    assert {"04-integracion-afip.pdf", "08-codigos-de-error.json", "01-arquitectura.md"} <= fuentes
    paginas_pdf = [d.metadata["page"] for d in docs if d.metadata["source"].endswith(".pdf")]
    assert paginas_pdf == [1, 2, 3]


def test_pdf_repara_codigos_cortados():
    pdf = " ".join(d.page_content for d in load_documents() if d.metadata["source"].endswith(".pdf"))
    assert "ERR-AFIP-102" in pdf and "ERR-\n" not in pdf


def test_metadatos_aptos_para_pinecone(chunks):
    """Pinecone no acepta valores nulos; las listas deben ser de strings."""
    for c in chunks:
        m = c.metadata
        assert {"doc_id", "title", "source", "category", "tags", "page", "chunk_id"} <= m.keys()
        assert all(v is not None for v in m.values())
        assert isinstance(m["page"], int) and all(isinstance(t, str) for t in m["tags"])
        assert c.page_content.strip()


def test_chunk_ids_unicos_y_estables(chunks):
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    otra_vez = split_documents(load_documents(), build_splitter(token_counter=approx_tokens))
    assert ids == [c.metadata["chunk_id"] for c in otra_vez]  # reingesta = mismos ids (sin duplicados)


def test_chunks_respetan_el_tamano(chunks):
    assert all(approx_tokens(c.page_content) <= config.CHUNK_SIZE_TOKENS for c in chunks)


# --- Recuperador --------------------------------------------------------------------
def test_tokenizador_normaliza_y_parte_codigos():
    toks = tokenize("Error ERR-AFIP-102 en la Autenticación")
    assert "err-afip-102" in toks and {"err", "afip", "102", "autenticacion"} <= set(toks)
    assert "la" not in toks and "en" not in toks


def test_hibrido_devuelve_top5_sin_duplicados(rag):
    docs = rag.retrieve("¿Qué significa el error ERR-AFIP-102?")
    ids = [d.metadata["chunk_id"] for d in docs]
    assert len(docs) == 5 and len(ids) == len(set(ids))


def test_bm25_encuentra_codigos_exactos(rag):
    docs = rag.retrieve("ERR-AFIP-102", mode="bm25")
    assert {"integracion-afip", "codigos-error"} & {d.metadata["doc_id"] for d in docs[:2]}


def test_consulta_vacia(rag):
    with pytest.raises(ValueError):
        rag.retrieve("   ")


# --- Métricas -----------------------------------------------------------------------
def test_metricas():
    ids = ["a", "b", "a", "c", "d"]
    assert precision_at_k(ids, {"a"}, 5) == 0.4
    assert recall_at_k(ids, "c", 5) == 1.0 and recall_at_k(ids, "z", 5) == 0.0
    assert reciprocal_rank(ids, "b") == 0.5 and reciprocal_rank(ids, "z") == 0.0


def test_evaluate_estructura(rag):
    golden = [{"id": "q", "pregunta": "ERR-AFIP-102", "documento_id_esperado": "integracion-afip",
               "documentos_relevantes": ["integracion-afip", "codigos-error"]}]
    r = evaluate(rag, golden, ["bm25", "hybrid"])
    assert set(r) == {"bm25", "hybrid"} and r["bm25"]["recall_at_k"] == 1.0


# --- Setup del índice ---------------------------------------------------------------
class FakePinecone:
    def __init__(self, existe=False, dimension=1536):
        self.existe, self.dimension, self.creado = existe, dimension, None

    def has_index(self, name):
        return self.existe

    def describe_index(self, name):
        return SimpleNamespace(dimension=self.dimension, metric="cosine", status={"ready": True})

    def create_index(self, **kwargs):
        self.creado, self.existe = kwargs, True

    def Index(self, name):
        return f"index:{name}"


def test_crea_el_indice_serverless_si_no_existe():
    from setup_index import ensure_index

    pc = FakePinecone(existe=False)
    assert ensure_index(pc) == f"index:{config.INDEX_NAME}"
    assert pc.creado["dimension"] == 1536 and pc.creado["metric"] == "cosine"
    assert pc.creado["spec"].cloud == "aws"


def test_detecta_mismatch_de_dimensiones():
    from setup_index import ensure_index

    with pytest.raises(SystemExit, match="Mismatch de dimensiones"):
        ensure_index(FakePinecone(existe=True, dimension=768))


# --- Prioridad a identificadores exactos -------------------------------------------
def test_detecta_identificadores_y_los_prioriza(rag):
    docs = rag.retrieve("¿Qué quiere decir el código OLA-RIB-0042?")  # modo por defecto: hybrid_exact
    assert docs[0].metadata["doc_id"] == "inventario"
    assert "OLA-RIB-0042" in docs[0].page_content


def test_sin_identificadores_es_igual_al_hibrido(rag):
    q = "¿Cuándo un pedido de la tienda se convierte en venta?"
    assert rag.exact_matches(q) == []
    assert [d.metadata["chunk_id"] for d in rag.retrieve(q)] == \
           [d.metadata["chunk_id"] for d in rag.retrieve(q, mode="hybrid")]
