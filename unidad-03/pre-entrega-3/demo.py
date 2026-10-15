"""
Pruebas end-to-end contra las APIs reales (OpenAI embeddings + Anthropic):
preguntas cuya respuesta está en los documentos y "preguntas trampa" que no.

    python demo.py
"""

import asyncio

from rag import get_rag_response

PREGUNTAS = [
    # Respuesta en los documentos
    ("en_documentos", "Se me rompió una bisagra del rack a los 4 meses. ¿Está en garantía y cómo hago el reclamo?"),
    ("en_documentos", "¿Puedo devolver un placar a medida si me arrepiento?"),
    # Preguntas trampa: la información NO está en los documentos
    ("trampa", "¿En cuántas cuotas sin interés puedo pagar con tarjeta de crédito?"),
    ("trampa", "¿Cuántos empleados tiene OLA Muebles y quién es el dueño?"),
]


async def main() -> None:
    # Las 4 consultas corren en paralelo (Unidad 1: asyncio.gather)
    resultados = await asyncio.gather(*(get_rag_response(p) for _, p in PREGUNTAS), return_exceptions=True)

    for (tipo, pregunta), r in zip(PREGUNTAS, resultados):
        print("\n" + "=" * 80)
        print(f"[{tipo}] {pregunta}")
        if isinstance(r, Exception):
            print(f"✗ Error: {type(r).__name__}: {r}")
            continue
        print(r.model_dump_json(indent=2))

        esperado = tipo == "en_documentos"
        estado = "OK" if r.encontrada_en_contexto == esperado else "REVISAR"
        print(f"→ {estado}: encontrada_en_contexto={r.encontrada_en_contexto} (esperado {esperado})")


if __name__ == "__main__":
    asyncio.run(main())
