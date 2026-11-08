"""Utilidades compartidas por los especialistas."""

import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph

import config


def texto_de(msg: BaseMessage) -> str:
    """Texto plano de un mensaje (Anthropic puede devolver una lista de bloques)."""
    if isinstance(msg.content, str):
        return msg.content
    return "".join(b if isinstance(b, str) else b.get("text", "") for b in msg.content
                   if isinstance(b, str) or (isinstance(b, dict) and b.get("type") == "text"))


def diagnostico(msg: BaseMessage) -> dict:
    """Metadatos de la respuesta del modelo (motivo de corte, tipos de bloque) para la traza."""
    bloques = ["str"] if isinstance(msg.content, str) else [
        b if isinstance(b, str) else b.get("type", "?") for b in msg.content]
    meta = getattr(msg, "response_metadata", {}) or {}
    return {"stop_reason": meta.get("stop_reason") or meta.get("finish_reason"), "bloques": bloques,
            "caracteres": len(texto_de(msg))}


def parse_json(contenido: Any) -> Any:
    try:
        return json.loads(contenido)
    except (TypeError, ValueError):
        return contenido


async def correr_especialista(agente: CompiledStateGraph, encargo: str) -> tuple[str, list[str], list[Any]]:
    """Ejecuta el ciclo ReAct de un especialista con SOLO su encargo (sin el historial del sistema).

    Devuelve (respuesta final, herramientas usadas en orden, resultados de las herramientas).
    """
    salida = await agente.ainvoke({"messages": [("user", encargo)]},
                                  {"recursion_limit": config.AGENT_RECURSION_LIMIT})
    mensajes: list[BaseMessage] = salida["messages"]
    herramientas = [tc["name"] for m in mensajes if isinstance(m, AIMessage) for tc in m.tool_calls]
    resultados = [parse_json(m.content) for m in mensajes if isinstance(m, ToolMessage)]
    respuesta = texto_de(mensajes[-1]).strip() or "(el especialista no devolvió texto)"
    return respuesta, herramientas, resultados


def pregunta_original(state: dict) -> str:
    """La consulta del usuario: el último mensaje humano de la conversación."""
    for m in reversed(state["messages"]):
        if m.type == "human":
            return texto_de(m)
    return ""
