"""
Mini-script de prueba asíncrono contra la API real de Anthropic.

Ejecutar desde la carpeta pre-entrega-2 (con ANTHROPIC_API_KEY en unidad-02/.env):
    python test_pipeline.py
"""

import asyncio
import time

from chain import logger, process_text

TEXTOS = {
    "log_error": (
        "ERROR 2026-10-14 03:12:44 api-gateway: psycopg2.OperationalError: FATAL: "
        "too many connections for role 'app'. Los pods de FastAPI en Kubernetes "
        "devuelven 503 en /checkout desde hace 20 minutos y Redis muestra 98% de memoria."
    ),
    "arquitectura": (
        "El frontend está hecho en Next.js y consume una API REST en Node.js con Express. "
        "Los datos viven en Supabase (PostgreSQL) y los archivos en un bucket de S3. "
        "El deploy se hace con GitHub Actions hacia Vercel."
    ),
    # Prueba de estrés: texto ambiguo, sin tecnologías claras ni impacto definido
    "ambiguo": "Ayer algo andaba lento, pero hoy parece que ya está mejor. Lo seguimos mirando.",
}


async def main() -> None:
    inicio = time.perf_counter()
    # Los tres textos se procesan en paralelo (Unidad 1: asyncio.gather)
    resultados = await asyncio.gather(
        *(process_text(texto) for texto in TEXTOS.values()),
        return_exceptions=True,
    )
    logger.info("Lote terminado en %.2fs", time.perf_counter() - inicio)

    for nombre, resultado in zip(TEXTOS, resultados):
        print(f"\n=== {nombre} ===")
        if isinstance(resultado, Exception):
            print(f"✗ {type(resultado).__name__}: {resultado}")
        else:
            print(resultado.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
