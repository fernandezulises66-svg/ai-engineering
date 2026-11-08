"""
Agente ReAct construido con LangGraph:

    START ──► agent ──(tools_condition)──► tools ──┐
                ▲  │                                │
                │  └──────── sin tool_calls ──► END │
                └───────────────────────────────────┘

- `agent`: el LLM con las herramientas vinculadas (bind_tools). Decide solo si
  responde o si pide una herramienta; no hay rutas if/else escritas a mano.
- `tools`: ejecuta las herramientas pedidas y devuelve el resultado como ToolMessage.
- `tools_condition`: si el último mensaje del modelo trae tool_calls va a `tools`;
  si no, termina. El ciclo agent ↔ tools se repite hasta que el modelo responde.
- El estado (lista de mensajes) se guarda en un checkpointer SQLite por thread_id.
"""

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, trim_messages
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

import config
from tools import TOOLS

logger = logging.getLogger("agente")

SYSTEM_PROMPT = """Sos el asistente interno del ERP de OLA Muebles. Respondés consultas sobre clientes y pedidos.

Reglas:
- Usá las herramientas para obtener datos: nunca inventes clientes, pedidos ni montos.
- Si una herramienta devuelve un `error`, leé la `sugerencia` y volvé a intentar de otra forma
  antes de rendirte. Si después de reintentar sigue faltando información, pedile al usuario
  la aclaración concreta que necesitás.
- Si una búsqueda devuelve varios clientes posibles, preguntá cuál es; no elijas al azar.
- Usá el contexto de la conversación: si el usuario dice "el último" o "ese cliente",
  referite a lo que ya se habló.
- Respondé en español rioplatense, breve. Los montos en pesos con separador de miles (por ejemplo, $14.500)."""


class AgentState(MessagesState):
    """Estado del grafo. Hereda `messages` con el reducer `add_messages`: cada nodo
    devuelve solo los mensajes nuevos y LangGraph los agrega al historial."""


def build_graph(llm: BaseChatModel, checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    llm_con_tools = llm.bind_tools(TOOLS)

    async def agent(state: AgentState) -> dict[str, list[BaseMessage]]:
        # El estado guarda TODO el historial, pero al modelo solo le mandamos los últimos
        # mensajes (sin cortar un par tool_call/tool_result) para que el contexto no crezca sin límite.
        recientes = trim_messages(state["messages"], max_tokens=config.MAX_CONTEXT_MESSAGES, token_counter=len,
                                  strategy="last", start_on="human", include_system=False, allow_partial=False)
        respuesta: AIMessage = await llm_con_tools.ainvoke([SystemMessage(SYSTEM_PROMPT), *recientes])
        if respuesta.tool_calls:
            logger.info("→ decide usar: %s", ", ".join(f"{tc['name']}({tc['args']})" for tc in respuesta.tool_calls))
        else:
            logger.info("→ responde sin herramientas")
        return {"messages": [respuesta]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    # handle_tool_errors: si una herramienta lanza una excepción (por ejemplo, argumentos inválidos),
    # el error vuelve al modelo como ToolMessage y puede corregirse en lugar de cortar la ejecución.
    graph.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)


def run_config(thread_id: str) -> dict[str, Any]:
    """Config de cada invocación: el thread_id identifica la conversación en el checkpointer
    y recursion_limit pone un techo de pasos para evitar bucles y costos inesperados."""
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": config.RECURSION_LIMIT}
