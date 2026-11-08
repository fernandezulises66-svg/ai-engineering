"""
Tests sin API keys: el LLM se reemplaza por un modelo "guionado" que devuelve
respuestas predefinidas (con o sin tool_calls). El grafo, las herramientas, la base
SQLite y el checkpointer son los reales.

    pytest -v
"""

import asyncio
import json
from itertools import count
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

import config
import db
from agent import build_graph
from runner import run_turn
from tools import buscar_cliente, buscar_pedidos, detalle_pedido

_ids = count(1)


def llamar(herramienta: str, /, **args: Any) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": herramienta, "args": args, "id": f"call_{next(_ids)}"}])


class ScriptedModel(BaseChatModel):
    """Devuelve en orden las respuestas del guion y guarda lo que recibió en cada llamada."""

    script: list[AIMessage]
    recibidos: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.recibidos.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])


@pytest.fixture(autouse=True)
def base_temporal(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "erp.db")
    asyncio.run(db.init_db(config.DB_PATH))
    return tmp_path


async def conversar(tmp_path, script, mensajes, thread="t1"):
    modelo = ScriptedModel(script=script, recibidos=[])
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "ckpt.sqlite")) as saver:
        graph = build_graph(modelo, saver)
        turnos = [await run_turn(graph, thread, m) for m in mensajes]
    return turnos, modelo


def resultados(turno):
    return [p["resultado"] for p in turno["pasos"] if p["tipo"] == "resultado_herramienta"]


# --- Herramientas -----------------------------------------------------------------
def test_herramientas_directas():
    clientes = json.loads(asyncio.run(buscar_cliente.ainvoke({"nombre": "gomez"})))
    assert [c["cliente_id"] for c in clientes["clientes"]] == [102, 103]  # sin acentos ni mayúsculas
    pedidos = json.loads(asyncio.run(buscar_pedidos.ainvoke({"cliente_id": 102})))
    assert pedidos["cantidad"] == 3 and pedidos["total"] == 14500 and pedidos["ultimo_pedido_id"] == 5042
    detalle = json.loads(asyncio.run(detalle_pedido.ainvoke({"pedido_id": 5042})))
    assert len(detalle["items"]) == 2


def test_errores_controlados_con_sugerencia():
    sin_cliente = json.loads(asyncio.run(buscar_pedidos.ainvoke({"cliente_id": 999})))
    assert "error" in sin_cliente and "buscar_cliente" in sin_cliente["sugerencia"]
    no_existe = json.loads(asyncio.run(buscar_cliente.ainvoke({"nombre": "Martín Gómez"})))
    assert no_existe["clientes"] == [] and "apellido" in no_existe["sugerencia"]


def test_los_docstrings_llegan_al_modelo():
    for t in (buscar_cliente, buscar_pedidos, detalle_pedido):
        assert len(t.description) > 150  # descripción detallada: es lo único que lee el LLM


# --- Grafo: ciclo ReAct -----------------------------------------------------------
def test_razonamiento_multipaso_dos_herramientas(base_temporal):
    script = [llamar("buscar_cliente", nombre="Martina Gómez"), llamar("buscar_pedidos", cliente_id=102),
              AIMessage(content="Martina Gómez tuvo 3 pedidos por un total de $14.500.")]
    (turno,), _ = asyncio.run(conversar(base_temporal, script, ["¿Cuántos pedidos tuvo Martina Gómez?"]))
    assert turno["llamadas_a_herramientas"] == 2
    assert resultados(turno)[1]["total"] == 14500
    assert turno["respuesta"].startswith("Martina Gómez tuvo 3 pedidos")


def test_ciclo_de_retorno_reintenta_tras_un_error(base_temporal):
    script = [llamar("buscar_cliente", nombre="Martín Gómez"), llamar("buscar_cliente", nombre="Gómez"),
              AIMessage(content="Encontré dos Gómez: Martina (City Bell) y Joaquín (Gonnet). ¿Cuál es?")]
    (turno,), _ = asyncio.run(conversar(base_temporal, script, ["¿Cuánto compró Martín Gómez?"]))
    primero, segundo = resultados(turno)
    assert "error" in primero and segundo["cantidad"] == 2
    assert turno["respuesta"].endswith("¿Cuál es?")


def test_excepcion_de_herramienta_vuelve_al_modelo(base_temporal):
    """Argumentos inválidos: ToolNode devuelve el error como mensaje y el agente puede corregir."""
    script = [llamar("buscar_pedidos", cliente_id="no-es-un-numero"), llamar("buscar_pedidos", cliente_id=102),
              AIMessage(content="Listo.")]
    (turno,), _ = asyncio.run(conversar(base_temporal, script, ["pedidos del 102"]))
    assert "error" in str(resultados(turno)[0]).lower()
    assert resultados(turno)[1]["cantidad"] == 3


def test_recursion_limit_corta_bucles(base_temporal):
    script = [llamar("buscar_cliente", nombre="Gómez") for _ in range(30)]
    (turno,), _ = asyncio.run(conversar(base_temporal, script, ["buscá en bucle"]))
    assert turno["pasos"][-1]["tipo"] == "error" and "límite" in turno["respuesta"]
    assert turno["llamadas_a_herramientas"] <= config.RECURSION_LIMIT


# --- Memoria y persistencia --------------------------------------------------------
def test_memoria_en_el_mismo_thread(base_temporal):
    script = [llamar("buscar_cliente", nombre="Martina Gómez"), llamar("buscar_pedidos", cliente_id=102),
              AIMessage(content="3 pedidos, $14.500."),
              llamar("detalle_pedido", pedido_id=5042), AIMessage(content="El último es el 5042.")]
    turnos, modelo = asyncio.run(conversar(base_temporal, script, ["¿Pedidos de Martina Gómez?", "¿Y el último?"]))
    contexto_turno_2 = modelo.recibidos[-1]
    assert any(isinstance(m, HumanMessage) and "Martina" in m.content for m in contexto_turno_2)
    assert resultados(turnos[1])[0]["pedido_id"] == 5042


def test_persistencia_entre_reinicios_y_aislamiento_por_thread(base_temporal):
    asyncio.run(conversar(base_temporal, [AIMessage(content="Hola Ulises.")], ["Me llamo Ulises."], thread="a"))
    # nuevo checkpointer sobre el mismo archivo = el programa se reinició
    _, modelo = asyncio.run(conversar(base_temporal, [AIMessage(content="Te llamás Ulises.")],
                                      ["¿Cómo me llamo?"], thread="a"))
    assert any("Me llamo Ulises" in str(m.content) for m in modelo.recibidos[0])
    _, otro = asyncio.run(conversar(base_temporal, [AIMessage(content="No sé.")], ["¿Cómo me llamo?"], thread="b"))
    assert not any("Ulises" in str(m.content) for m in otro.recibidos[0])  # otro thread, otra memoria


def test_el_contexto_enviado_al_modelo_esta_acotado(base_temporal):
    mensajes = [f"mensaje {i}" for i in range(15)]  # 15 turnos = 30 mensajes en el estado
    script = [AIMessage(content=f"ok {i}") for i in range(15)]
    _, modelo = asyncio.run(conversar(base_temporal, script, mensajes))
    ultimo = modelo.recibidos[-1]
    assert len(ultimo) <= config.MAX_CONTEXT_MESSAGES + 1  # + el system prompt
    assert "mensaje 14" in ultimo[-1].content
