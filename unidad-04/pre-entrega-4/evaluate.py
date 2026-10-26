"""
Evaluación del recuperador con un Golden Set: Precision@k, Recall@k y MRR,
comparando BM25, búsqueda vectorial, híbrida (RRF) e híbrida con prioridad a
coincidencias exactas de identificadores.

    python evaluate.py                 # los 4 modos contra Pinecone
    python evaluate.py --modes bm25    # solo BM25 (no necesita Pinecone ni OpenAI)

Definiciones (k = 5, a nivel de documento fuente):
  - Recall@5: 1 si el documento esperado aparece entre los 5 chunks recuperados, 0 si no.
  - Precision@5: proporción de los 5 chunks que vienen de un documento relevante.
  - MRR: 1 / posición del primer chunk del documento esperado (0 si no aparece).
"""

import argparse
import json
import logging
from pathlib import Path

import config
from rag_system import RAGSystem

GOLDEN_SET = config.BASE_DIR / "golden_set.json"
RESULTS = config.BASE_DIR / "resultados_evaluacion.json"


def precision_at_k(doc_ids: list[str], relevantes: set[str], k: int) -> float:
    return sum(d in relevantes for d in doc_ids[:k]) / k


def recall_at_k(doc_ids: list[str], esperado: str, k: int) -> float:
    return 1.0 if esperado in doc_ids[:k] else 0.0


def reciprocal_rank(doc_ids: list[str], esperado: str) -> float:
    return next((1 / i for i, d in enumerate(doc_ids, start=1) if d == esperado), 0.0)


def evaluate(rag: RAGSystem, golden: list[dict], modes: list[str], k: int = config.TOP_K) -> dict:
    resultados = {}
    for mode in modes:
        filas = []
        for q in golden:
            docs = rag.retrieve(q["pregunta"], mode=mode)
            doc_ids = [d.metadata["doc_id"] for d in docs]
            filas.append({
                "id": q["id"],
                "pregunta": q["pregunta"],
                "esperado": q["documento_id_esperado"],
                "recuperados": [d.metadata["chunk_id"] for d in docs],
                "precision": precision_at_k(doc_ids, set(q["documentos_relevantes"]), k),
                "recall": recall_at_k(doc_ids, q["documento_id_esperado"], k),
                "rr": reciprocal_rank(doc_ids, q["documento_id_esperado"]),
            })
        n = len(filas)
        resultados[mode] = {
            "precision_at_k": sum(f["precision"] for f in filas) / n,
            "recall_at_k": sum(f["recall"] for f in filas) / n,
            "mrr": sum(f["rr"] for f in filas) / n,
            "detalle": filas,
        }
    return resultados


def max_precision(rag: RAGSystem, golden: list[dict], k: int) -> dict[str, float]:
    """Techo de Precision@k: si un documento tiene menos de k chunks, nunca puede llegar a 1."""
    chunks_por_doc: dict[str, int] = {}
    for d in rag.bm25.docs:
        chunks_por_doc[d.metadata["doc_id"]] = chunks_por_doc.get(d.metadata["doc_id"], 0) + 1
    return {q["id"]: min(k, sum(chunks_por_doc.get(r, 0) for r in q["documentos_relevantes"])) / k for q in golden}


def print_report(resultados: dict, techo: dict[str, float], k: int) -> None:
    nombres = {"bm25": "BM25 (léxico)", "vector": "Vectorial (Pinecone)", "hybrid": "Híbrido (RRF)",
               "hybrid_exact": "Híbrido + exactos"}
    print(f"\n{'=' * 72}\nREPORTE DE EVALUACIÓN · {len(next(iter(resultados.values()))['detalle'])} preguntas · k = {k}\n{'=' * 72}")
    print(f"{'Recuperador':<24}{'Precision@' + str(k):>14}{'Recall@' + str(k):>12}{'MRR':>10}")
    for mode, r in resultados.items():
        print(f"{nombres[mode]:<24}{r['precision_at_k']:>14.2f}{r['recall_at_k']:>12.2f}{r['mrr']:>10.2f}")
    print(f"\nTecho de Precision@{k} promedio (por cantidad de chunks relevantes): {sum(techo.values()) / len(techo):.2f}")

    print("\nDetalle por pregunta (P = precision, R = recall, posición del doc esperado):")
    modos = list(resultados)
    print(f"{'':<4}" + "".join(f"{nombres[m]:>22}" for m in modos))
    for i, fila in enumerate(resultados[modos[0]]["detalle"]):
        celdas = []
        for m in modos:
            f = resultados[m]["detalle"][i]
            pos = f"#{round(1 / f['rr'])}" if f["rr"] else "—"
            celdas.append(f"P={f['precision']:.1f} R={f['recall']:.0f} {pos:>3}")
        print(f"{fila['id']:<4}" + "".join(f"{c:>22}" for c in celdas) + f"   {fila['pregunta'][:40]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["bm25", "vector", "hybrid", "hybrid_exact"],
                        choices=["bm25", "vector", "hybrid", "hybrid_exact"])
    args = parser.parse_args()
    logging.getLogger("rag").setLevel(logging.WARNING)

    golden = json.loads(GOLDEN_SET.read_text(encoding="utf-8"))
    if args.modes == ["bm25"]:
        from langchain_core.embeddings import DeterministicFakeEmbedding
        from langchain_core.vectorstores import InMemoryVectorStore

        from ingest import load_chunks
        rag = RAGSystem(load_chunks(), InMemoryVectorStore(DeterministicFakeEmbedding(size=8)))
    else:
        rag = RAGSystem.from_pinecone()

    resultados = evaluate(rag, golden, args.modes)
    techo = max_precision(rag, golden, config.TOP_K)
    print_report(resultados, techo, config.TOP_K)
    RESULTS.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResultados completos guardados en {RESULTS.name}")


if __name__ == "__main__":
    main()
