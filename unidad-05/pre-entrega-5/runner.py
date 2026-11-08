"""Ejecuta un turno del agente y registra la traza ReAct (decisión → herramienta → resultado → respuesta)."""

import json
import logging
from typing import Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

import config
from agent import run_config

logger = logging.getLogger("runner")


class Paso(TypedDict, total=False):
    tipo: str            # "decision_herramienta" | "resultado_herramienta" | "respuesta" | "error"
    herramienta: str
    argumentos: dict[str, Any]
    resultado: Any
    texto: str


class Turno(TypedDict):
    thread_id: str
    usuario: str
    pasos: list[Paso]
    respuesta: str
    llamadas_a_herramientas: int


def _texto(msg: AIMessage) -> str:
    if isinstance(msg.content, str):
        return msg.content
    return "".join(b.get("text", "") for b in msg.content if isinstance(b, dict) and b.get("type") == "text")


def _parse(contenido: Any) -> Any:
    try:
        return json.loads(contenido)
    except (TypeError, ValueError):
        return contenido


async def run_turn(graph: CompiledStateGraph, thread_id: str, mensaje: str) -> Turno:
    """Envía un mensaje en un thread y devuelve la traza completa del turno."""
    pasos: list[Paso] = []
    respuesta = ""
    try:
        # stream_mode="updates": un evento por nodo ejecutado, con los mensajes nuevos que produjo
        async for evento in graph.astream({"messages": [HumanMessage(mensaje)]}, run_config(thread_id),
                                          stream_mode="updates"):
            for nodo, update in evento.items():
                for msg in (update or {}).get("messages", []):
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        for tc in msg.tool_calls:
                            pasos.append({"tipo": "decision_herramienta", "herramienta": tc["name"],
                                          "argumentos": tc["args"]})
                    elif isinstance(msg, AIMessage):
                        respuesta = _texto(msg)
                        pasos.append({"tipo": "respuesta", "texto": respuesta})
                    elif isinstance(msg, ToolMessage):
                        pasos.append({"tipo": "resultado_herramienta", "herramienta": msg.name or nodo,
                                      "resultado": _parse(msg.content)})
    except GraphRecursionError:
        respuesta = (f"No pude terminar el razonamiento dentro del límite de {config.RECURSION_LIMIT} pasos. "
                     "¿Podés reformular o dividir la consulta?")
        pasos.append({"tipo": "error", "texto": respuesta})
        logger.warning("Se alcanzó el recursion_limit (%d) en el thread %s", config.RECURSION_LIMIT, thread_id)

    return {
        "thread_id": thread_id,
        "usuario": mensaje,
        "pasos": pasos,
        "respuesta": respuesta,
        "llamadas_a_herramientas": sum(p["tipo"] == "decision_herramienta" for p in pasos),
    }


def format_turn(turno: Turno) -> str:
    """Traza legible, en el formato del ejemplo de la consigna."""
    lineas = [f'[thread {turno["thread_id"]}] Usuario: "{turno["usuario"]}"']
    for p in turno["pasos"]:
        if p["tipo"] == "decision_herramienta":
            args = ", ".join(f"{k}={v!r}" for k, v in p["argumentos"].items())
            lineas.append(f"  → El agente decide usar la herramienta: {p['herramienta']}({args})")
        elif p["tipo"] == "resultado_herramienta":
            lineas.append(f"  → La herramienta devuelve: {json.dumps(p['resultado'], ensure_ascii=False)}")
        elif p["tipo"] == "error":
            lineas.append(f"  ✗ {p['texto']}")
    lineas.append(f'  Respuesta: "{turno["respuesta"]}"')
    lineas.append(f"  (llamadas a herramientas en este turno: {turno['llamadas_a_herramientas']})")
    return "\n".join(lineas)
