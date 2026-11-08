"""
Grafo del orquestador multi-agente (topología jerárquica: supervisor + especialistas).

    START → supervisor ─┬─► investigador ─┐
               ▲        ├─► analista ─────┤
               └────────┼─────────────────┘
                        └─► sintetizador → END
"""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

import config
from agents import analyst_agent, research_agent, supervisor, synthesizer
from knowledge import KnowledgeBase, get_kb
from state import OrchestratorState

ROLES = ("supervisor", "investigador", "analista", "sintetizador")


def build_graph(llms: dict[str, BaseChatModel] | None = None, kb: KnowledgeBase | None = None) -> CompiledStateGraph:
    """`llms` permite usar un modelo distinto por rol (por defecto, el mismo para todos)."""
    llms = llms or {}
    base = None if all(r in llms for r in ROLES) else config.get_llm()
    modelo = {r: llms.get(r, base) for r in ROLES}

    graph = StateGraph(OrchestratorState)
    graph.add_node("supervisor", supervisor.crear_nodo(modelo["supervisor"]))
    graph.add_node("investigador", research_agent.crear_nodo(modelo["investigador"], kb or get_kb()))
    graph.add_node("analista", analyst_agent.crear_nodo(modelo["analista"]))
    graph.add_node("sintetizador", synthesizer.crear_nodo(modelo["sintetizador"]))

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", supervisor.route_supervisor,
                                {"investigador": "investigador", "analista": "analista", "sintetizador": "sintetizador"})
    graph.add_edge("investigador", "supervisor")   # todo especialista reporta al supervisor
    graph.add_edge("analista", "supervisor")
    graph.add_edge("sintetizador", END)
    return graph.compile()


def estado_inicial(pregunta: str) -> dict[str, Any]:
    return {"messages": [("user", pregunta)], "contribuciones": [], "decisiones": [], "delegaciones": 0,
            "task_completed": False, "instruccion": "", "respuesta_final": ""}


def run_config() -> dict[str, Any]:
    return {"recursion_limit": config.RECURSION_LIMIT}
