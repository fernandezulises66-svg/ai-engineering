"""
Prueba de ejecución que genera la traza ReAct de ejemplo (trazas/traza_ejemplo.json y .log).

    python demo.py

Escenarios:
  1. Razonamiento multi-paso: el agente encadena buscar_cliente → buscar_pedidos.
  2. Memoria: en el mismo thread_id, "¿y el último?" se resuelve con el contexto previo.
  3. Ciclo de retorno: una búsqueda sin resultados → reintento → ambigüedad → pide aclaración.
  4. Persistencia: se cierra el checkpointer, se abre uno nuevo (como si se reiniciara el
     programa) y el agente sigue recordando la conversación del escenario 1.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

import config
import db
from agent import build_graph
from runner import Turno, format_turn, run_turn

ESCENARIOS: list[tuple[str, str, list[str]]] = [
    ("multi-paso + memoria", "clientes", [
        "¿Cuántos pedidos tuvo Martina Gómez y cuál fue el total?",
        "¿Y el último?",
    ]),
    ("ciclo de retorno (reintento + aclaración)", "reintento", [
        "Necesito saber cuánto compró Martín Gómez.",
        "El de Gonnet.",
    ]),
]


async def main() -> None:
    await db.init_db()
    sufijo = uuid.uuid4().hex[:6]  # threads nuevos en cada corrida
    llm = config.get_llm()
    turnos: list[tuple[str, Turno]] = []
    inicio = time.perf_counter()

    async with AsyncSqliteSaver.from_conn_string(str(config.CHECKPOINT_DB)) as saver:
        graph = build_graph(llm, saver)
        for nombre, thread, mensajes in ESCENARIOS:
            for m in mensajes:
                turno = await run_turn(graph, f"{thread}-{sufijo}", m)
                turnos.append((nombre, turno))
                print(format_turn(turno), "\n")

    # Escenario 4: checkpointer nuevo (equivale a reiniciar el proceso) y mismo thread_id
    async with AsyncSqliteSaver.from_conn_string(str(config.CHECKPOINT_DB)) as saver:
        graph = build_graph(llm, saver)
        turno = await run_turn(graph, f"clientes-{sufijo}",
                               "Retomando lo de antes: ¿de qué cliente estábamos hablando y qué incluía su último pedido?")
        turnos.append(("persistencia tras reiniciar el checkpointer", turno))
        print(format_turn(turno), "\n")

    config.TRAZAS_DIR.mkdir(exist_ok=True)
    traza = {
        "generada": datetime.now().isoformat(timespec="seconds"),
        "proveedor": config.LLM_PROVIDER,
        "recursion_limit": config.RECURSION_LIMIT,
        "duracion_segundos": round(time.perf_counter() - inicio, 1),
        "turnos": [{"escenario": nombre, **t} for nombre, t in turnos],
    }
    (config.TRAZAS_DIR / "traza_ejemplo.json").write_text(json.dumps(traza, ensure_ascii=False, indent=2), encoding="utf-8")
    (config.TRAZAS_DIR / "traza_ejemplo.log").write_text(
        "\n\n".join(f"### Escenario: {n}\n{format_turn(t)}" for n, t in turnos) + "\n", encoding="utf-8")
    print(f"Trazas guardadas en {config.TRAZAS_DIR.name}/traza_ejemplo.json y .log")


if __name__ == "__main__":
    asyncio.run(main())
