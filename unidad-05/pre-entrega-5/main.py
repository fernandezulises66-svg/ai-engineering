"""
Chat interactivo con el agente. La conversación se guarda en SQLite por thread_id:
si cerrás el programa y lo volvés a abrir con el mismo --thread, el agente recuerda.

    python main.py --thread mi-sesion
    python main.py --thread mi-sesion --verbose    # muestra la traza ReAct de cada turno
"""

import argparse
import asyncio

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

import config
import db
from agent import build_graph
from runner import format_turn, run_turn


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread", default="sesion-local", help="identificador de la conversación")
    parser.add_argument("--verbose", action="store_true", help="mostrar la traza de razonamiento")
    args = parser.parse_args()

    await db.init_db()
    async with AsyncSqliteSaver.from_conn_string(str(config.CHECKPOINT_DB)) as saver:
        graph = build_graph(config.get_llm(), saver)
        print(f"Agente ERP OLA Muebles · thread '{args.thread}' · escribí 'salir' para terminar\n")
        while True:
            mensaje = (await asyncio.to_thread(input, "Vos: ")).strip()
            if mensaje.lower() in {"salir", "exit", "quit"}:
                break
            if not mensaje:
                continue
            turno = await run_turn(graph, args.thread, mensaje)
            print(format_turn(turno) if args.verbose else f"Agente: {turno['respuesta']}", "\n")


if __name__ == "__main__":
    asyncio.run(main())
