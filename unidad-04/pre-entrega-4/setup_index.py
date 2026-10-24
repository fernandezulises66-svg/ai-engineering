"""
Inicialización de la infraestructura: verifica si el índice existe y, si no, lo
crea en modo Serverless. Si ya existe, comprueba que su dimensión y métrica
coincidan con el modelo de embeddings (evita el "mismatch de dimensiones").

    python setup_index.py
"""

import logging
import time

import config

logger = logging.getLogger("setup")


def ensure_index(pc=None, wait_seconds: int = 120):
    from pinecone import ServerlessSpec

    pc = pc or config.get_pinecone()
    name = config.INDEX_NAME

    if pc.has_index(name):
        desc = pc.describe_index(name)
        logger.info("El índice '%s' ya existe (dimensión %s, métrica %s)", name, desc.dimension, desc.metric)
        if desc.dimension != config.EMBEDDING_DIM:
            raise SystemExit(
                f"Mismatch de dimensiones: el índice '{name}' tiene {desc.dimension} y el modelo "
                f"'{config.EMBEDDING_MODEL}' genera {config.EMBEDDING_DIM}. Usá otro INDEX_NAME o borrá el índice."
            )
        if not str(desc.metric).lower().endswith(config.METRIC):
            logger.warning("El índice usa la métrica %s y se esperaba %s", desc.metric, config.METRIC)
        return pc.Index(name)

    logger.info("Creando índice serverless '%s' (%d dimensiones, %s, %s/%s)...", name, config.EMBEDDING_DIM,
                config.METRIC, config.PINECONE_CLOUD, config.PINECONE_REGION)
    pc.create_index(
        name=name,
        dimension=config.EMBEDDING_DIM,
        metric=config.METRIC,
        spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
        deletion_protection="disabled",
        tags={"proyecto": "ai-engineering", "dataset": "ola-erp"},
    )
    inicio = time.time()
    while not pc.describe_index(name).status.get("ready", False):
        if time.time() - inicio > wait_seconds:
            raise SystemExit(f"El índice '{name}' no quedó listo en {wait_seconds}s")
        time.sleep(2)
    logger.info("Índice '%s' listo", name)
    return pc.Index(name)


if __name__ == "__main__":
    index = ensure_index()
    stats = index.describe_index_stats()
    logger.info("Vectores en el índice: %s · namespaces: %s", stats.total_vector_count, list(stats.namespaces or {}))
