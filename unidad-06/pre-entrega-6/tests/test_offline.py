"""
Tests sin API keys: cada rol usa un modelo "guionado" que devuelve respuestas predefinidas.
El grafo, el supervisor (con sus salvaguardas), las herramientas y la búsqueda BM25 sobre
el manual son los reales.

    pytest -v
"""

import asyncio
import json
from itertools import count
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import config
from agents.analyst_agent import aplicar_porcentaje, calcular, dias_entre_fechas, sumar_dias
from agents.research_agent import crear_herramientas
from graph import build_graph
from knowledge import LocalKB
from runner import PREGUNTA_DEMO, ejecutar

_ids = count(1)


def llamar(herramienta: str, /, **args: Any) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": herramienta, "args": args, "id": f"call_{next(_ids)}"}])


def decidir(siguiente: str, instruccion: str = "", motivo: str = "ok") -> AIMessage:
    return AIMessage(json.dumps({"siguiente": siguiente, "instruccion": instruccion, "motivo": motivo}))


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


def modelos(supervisor, investigador=(), analista=(), sintetizador=("Respuesta final.",)) -> dict[str, ScriptedModel]:
    a_msg = lambda xs: [x if isinstance(x, AIMessage) else AIMessage(x) for x in xs]
    return {"supervisor": ScriptedModel(script=a_msg(supervisor), recibidos=[]),
            "investigador": ScriptedModel(script=a_msg(investigador), recibidos=[]),
            "analista": ScriptedModel(script=a_msg(analista), recibidos=[]),
            "sintetizador": ScriptedModel(script=a_msg(sintetizador), recibidos=[])}


def correr(llms, pregunta=PREGUNTA_DEMO) -> dict[str, Any]:
    return asyncio.run(ejecutar(build_graph(llms, LocalKB()), pregunta, mostrar=None))


def resultado(tool, **args) -> dict:
    return json.loads(tool.invoke(args))


# --- Herramientas ---------------------------------------------------------------------
def test_herramientas_del_analista():
    assert resultado(calcular, expresion="1180000 * (1 - 0.18)")["resultado"] == 967600
    assert resultado(dias_entre_fechas, desde="28/10/2026", hasta="2026-11-10")["dias_corridos"] == 13
    assert resultado(sumar_dias, fecha="28/10/2026", dias=15)["resultado"] == "2026-11-12"
    assert resultado(aplicar_porcentaje, monto=967600, porcentaje=50, operacion="porcion")["resultado"] == 483800
    assert resultado(aplicar_porcentaje, monto=1000, porcentaje=10, operacion="recargo")["resultado"] == 1100


def test_calcular_no_ejecuta_codigo_arbitrario():
    assert "error" in resultado(calcular, expresion="__import__('os').system('echo hola')")
    assert "error" in resultado(calcular, expresion="1 / 0")


def test_busqueda_en_el_manual_encuentra_las_reglas():
    buscar = crear_herramientas(LocalKB())[0]
    assert len(buscar.description) > 200   # docstring descriptivo para que el modelo sepa cuándo usarla
    validez = json.loads(asyncio.run(buscar.ainvoke({"consulta": "validez de un presupuesto"})))
    assert any("15 días" in f["texto"] and f["fuente"] == "03-api-ventas.md" for f in validez["fragmentos"])
    sin_nada = json.loads(asyncio.run(buscar.ainvoke({"consulta": "zzzz qqqq"})))
    assert sin_nada["fragmentos"] == [] and "sugerencia" in sin_nada


# --- Flujo de delegación ----------------------------------------------------------------
def flujo_completo() -> dict[str, ScriptedModel]:
    return modelos(
        supervisor=[decidir("investigador", "Buscá validez del presupuesto, seña y quién aprueba descuentos."),
                    decidir("analista", "Calculá vigencia, precio con descuento y seña."),
                    decidir("FINALIZAR", motivo="La rúbrica se cumple.")],
        investigador=[llamar("buscar_manual", consulta="validez de un presupuesto"),
                      llamar("buscar_manual", consulta="aprobar descuentos mayores al 15 %"),
                      "- Presupuesto: 15 días [03-api-ventas.md]\n- Seña 50 % [03-api-ventas.md]\n"
                      "- Descuentos > 15 %: gerencia [02-autenticacion-y-roles.md]"],
        analista=[llamar("dias_entre_fechas", desde="2026-10-28", hasta="2026-11-10"),
                  llamar("aplicar_porcentaje", monto=1180000, porcentaje=18, operacion="descuento"),
                  llamar("aplicar_porcentaje", monto=967600, porcentaje=50, operacion="porcion"),
                  "Vigente (13 de 15 días). Precio: $967.600. Seña: $483.800."],
        sintetizador=["Sigue vigente; queda en $967.600, seña de $483.800; lo aprueba gerencia."])


def test_supervisor_delega_investigador_luego_analista_y_cierra():
    llms = flujo_completo()
    traza = correr(llms)
    assert traza["recorrido"] == ["supervisor", "investigador", "supervisor", "analista", "supervisor", "sintetizador"]
    assert traza["task_completed"] and "$967.600" in traza["respuesta_final"]
    assert traza["delegaciones"] == 2

    inv, ana = traza["contribuciones"]
    assert (inv["agente"], inv["ronda"], inv["herramientas"]) == ("investigador", 1, ["buscar_manual"] * 2)
    assert {"03-api-ventas.md", "02-autenticacion-y-roles.md"} <= set(inv["fuentes"])
    assert (ana["agente"], ana["ronda"]) == ("analista", 2)
    assert ana["herramientas"] == ["dias_entre_fechas", "aplicar_porcentaje", "aplicar_porcentaje"]
    assert [d["siguiente"] for d in traza["decisiones"]] == ["investigador", "analista", "sintetizador"]


def test_contexto_acotado_por_especialista():
    llms = flujo_completo()
    correr(llms)
    # El investigador recibe solo su prompt + el encargo (no el historial ni la metadata del sistema)
    primera_inv = llms["investigador"].recibidos[0]
    assert [m.type for m in primera_inv] == ["system", "human"]
    assert "Encargo del supervisor" in primera_inv[1].content and "RÚBRICA" not in primera_inv[1].content
    # El analista recibe los hallazgos del investigador dentro de su encargo
    primera_ana = llms["analista"].recibidos[0]
    assert [m.type for m in primera_ana] == ["system", "human"] and "15 días" in primera_ana[1].content
    # El supervisor ve la pizarra con quién aportó qué
    ultima_sup = llms["supervisor"].recibidos[-1][1].content
    assert "INVESTIGADOR" in ultima_sup and "ANALISTA" in ultima_sup and "dias_entre_fechas" in ultima_sup


def test_supervisor_pide_refinar_al_analista():
    llms = modelos(
        supervisor=[decidir("investigador", "Buscá las reglas."),
                    decidir("analista", "Calculá todo."),
                    decidir("analista", "Falta la seña: calculala con aplicar_porcentaje.", "Criterio 3 no se cumple."),
                    decidir("FINALIZAR")],
        investigador=["- Seña 50 % [03-api-ventas.md]"],
        analista=["Precio: $967.600.",
                  llamar("aplicar_porcentaje", monto=967600, porcentaje=50, operacion="porcion"),
                  "Precio: $967.600. Seña: $483.800."])
    traza = correr(llms)
    analisis = [c for c in traza["contribuciones"] if c["agente"] == "analista"]
    assert [c["ronda"] for c in analisis] == [2, 3]
    assert analisis[1]["instruccion"].startswith("Falta la seña") and analisis[1]["herramientas"] == ["aplicar_porcentaje"]
    assert traza["recorrido"][-1] == "sintetizador"


def test_tope_de_revisiones_evita_el_supervisor_infinito():
    # El supervisor insiste siempre con el investigador: la salvaguarda corta el bucle
    llms = modelos(supervisor=[decidir("investigador", "Buscá de nuevo.")] * 10,
                   investigador=["- nada nuevo"] * 10)
    traza = correr(llms)
    investigaciones = [c for c in traza["contribuciones"] if c["agente"] == "investigador"]
    assert len(investigaciones) == config.MAX_REVISIONES + 1
    assert traza["decisiones"][-1]["origen"] == "regla" and traza["recorrido"][-1] == "sintetizador"


def test_tope_de_delegaciones(monkeypatch):
    monkeypatch.setattr(config, "MAX_DELEGACIONES", 2)
    llms = modelos(supervisor=[decidir("investigador", "a"), decidir("analista", "b"), decidir("investigador", "c")],
                   investigador=["- dato"], analista=["- cuenta"])
    traza = correr(llms)
    assert traza["delegaciones"] == 2
    assert traza["decisiones"][-1] == {"ronda": 3, "siguiente": "sintetizador", "origen": "regla",
                                       "motivo": "Se alcanzó el tope de 2 delegaciones: se cierra con lo disponible."}


def test_respuesta_invalida_del_supervisor_usa_regla_por_defecto():
    llms = modelos(supervisor=["no sé", "sigo sin saber", decidir("FINALIZAR")],
                   investigador=["- Presupuesto: 15 días [03-api-ventas.md]"])
    traza = correr(llms)
    assert traza["decisiones"][0]["origen"] == "regla" and traza["decisiones"][0]["siguiente"] == "investigador"
    assert traza["recorrido"] == ["supervisor", "investigador", "supervisor", "sintetizador"]


def test_diagrama_del_grafo():
    mermaid = build_graph(flujo_completo(), LocalKB()).get_graph().draw_mermaid()
    for nodo in ("supervisor", "investigador", "analista", "sintetizador"):
        assert nodo in mermaid
    assert "-.->" in mermaid   # aristas condicionales desde el supervisor
