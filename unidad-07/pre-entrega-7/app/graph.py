"""
Orquestador multi-agente de la pre-entrega 6 + Human-in-the-loop + checkpoints en Redis.

    START → supervisor ─┬─► investigador ─┐
               ▲        ├─► analista ─────┤
               └────────┼─────────────────┘
                        └─► sintetizador → evaluar_riesgo ─┬────────────────────────────► END
                                                           └► aprobacion_humana ─┬► ejecutar_accion → END
                                                               (interrupt)        └──────────────────► END
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app import config, hitl
from app.agents import analyst_agent, research_agent, supervisor, synthesizer
from app.knowledge import KnowledgeBase, get_kb
from app.state import OrchestratorState

ROLES = ("supervisor", "investigador", "analista", "sintetizador")


def build_graph(checkpointer: BaseCheckpointSaver, registrar: hitl.Registrador,
                llms: dict[str, BaseChatModel] | None = None, kb: KnowledgeBase | None = None) -> CompiledStateGraph:
    """`checkpointer` es obligatorio: sin él, `interrupt()` no puede pausar y reanudar el grafo."""
    llms = llms or {}
    base = None if all(r in llms for r in ROLES) else config.get_llm()
    modelo = {r: llms.get(r, base) for r in ROLES}

    graph = StateGraph(OrchestratorState)
    graph.add_node("supervisor", supervisor.crear_nodo(modelo["supervisor"]))
    graph.add_node("investigador", research_agent.crear_nodo(modelo["investigador"], kb or get_kb()))
    graph.add_node("analista", analyst_agent.crear_nodo(modelo["analista"]))
    graph.add_node("sintetizador", synthesizer.crear_nodo(modelo["sintetizador"]))
    graph.add_node("evaluar_riesgo", hitl.evaluar_riesgo)
    graph.add_node("aprobacion_humana", hitl.aprobacion_humana)
    graph.add_node("ejecutar_accion", hitl.crear_nodo_ejecutar(registrar))

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", supervisor.route_supervisor,
                                {"investigador": "investigador", "analista": "analista", "sintetizador": "sintetizador"})
    graph.add_edge("investigador", "supervisor")
    graph.add_edge("analista", "supervisor")
    graph.add_edge("sintetizador", "evaluar_riesgo")
    graph.add_conditional_edges("evaluar_riesgo", hitl.route_riesgo, {"aprobacion_humana": "aprobacion_humana", END: END})
    graph.add_conditional_edges("aprobacion_humana", hitl.route_aprobacion, {"ejecutar_accion": "ejecutar_accion", END: END})
    graph.add_edge("ejecutar_accion", END)
    return graph.compile(checkpointer=checkpointer)


@asynccontextmanager
async def redis_checkpointer(url: str = config.REDIS_URL) -> AsyncIterator[BaseCheckpointSaver]:
    """AsyncRedisSaver (langgraph-checkpoint-redis). Necesita Redis 8+ (trae RedisJSON y RediSearch)."""
    from langgraph.checkpoint.redis.aio import AsyncRedisSaver

    async with AsyncRedisSaver.from_conn_string(url) as saver:
        await saver.asetup()   # crea los índices la primera vez
        yield saver


def estado_inicial(pregunta: str) -> dict[str, Any]:
    return {"messages": [("user", pregunta)], "contribuciones": [], "decisiones": [], "delegaciones": 0,
            "task_completed": False, "instruccion": "", "respuesta_final": "", "es_critica": False,
            "accion_propuesta": None, "aprobacion": None, "accion_resultado": ""}


def run_config(job_id: str, tipo: str) -> dict[str, Any]:
    """thread_id = job_id: el checkpoint de Redis es la "memoria" de ese job entre la pausa y la reanudación."""
    return {"configurable": {"thread_id": job_id}, "recursion_limit": config.RECURSION_LIMIT,
            "run_name": f"orquestador.{tipo}", "tags": ["pe7", tipo], "metadata": {"job_id": job_id}}
