"""
Corre el orquestador por consola y guarda la traza del flujo de delegación.

    python main.py                       # pregunta de demostración
    python main.py "tu pregunta"         # otra consulta
    python main.py --diagrama            # regenera grafo.mmd y grafo.png
"""

import argparse
import asyncio

import config
from graph import build_graph
from runner import PREGUNTA_DEMO, ejecutar, guardar


def exportar_diagrama() -> None:
    graph = build_graph()
    mermaid = graph.get_graph().draw_mermaid()
    (config.BASE_DIR / "grafo.mmd").write_text(mermaid, encoding="utf-8")
    try:
        (config.BASE_DIR / "grafo.png").write_bytes(graph.get_graph().draw_mermaid_png())  # usa mermaid.ink
        print("Diagrama guardado en grafo.mmd y grafo.png")
    except Exception as e:  # sin internet o servicio caído: queda el .mmd
        print(f"Diagrama guardado en grafo.mmd (no se pudo generar el PNG: {e})")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pregunta", nargs="?", default=PREGUNTA_DEMO)
    parser.add_argument("--diagrama", action="store_true", help="exportar el diagrama del grafo")
    args = parser.parse_args()
    if args.diagrama:
        exportar_diagrama()
        return

    print(f"Pregunta: {args.pregunta}\n")
    traza = await ejecutar(build_graph(), args.pregunta)
    guardar(traza)
    print(f"Recorrido: {' → '.join(traza['recorrido'])} → END  ({traza['duracion_segundos']} s)")
    print("Traza guardada en trazas/traza_ejemplo.json y .log")


if __name__ == "__main__":
    asyncio.run(main())
