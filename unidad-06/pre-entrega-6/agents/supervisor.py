"""
Supervisor: el "cerebro" del orquestador. En cada ronda lee la pregunta y la pizarra de
contribuciones y decide quién interviene ahora (investigador o analista) o si la tarea está
completa y se pasa a la síntesis final. Valida los aportes con una rúbrica antes de cerrar.

Salvaguardas para evitar el "supervisor infinito" (no dependen del modelo):
- tope de delegaciones totales (MAX_DELEGACIONES);
- tope de revisiones por especialista (MAX_REVISIONES);
- si la respuesta del modelo no se puede interpretar, una regla determinista decide.
"""

import json
import logging
import re
from collections import Counter
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

import config
from agents.common import pregunta_original, texto_de
from state import Contribucion, Destino, OrchestratorState

logger = logging.getLogger("supervisor")

PROMPT = """Sos el Supervisor de un equipo que responde consultas sobre el ERP de OLA Muebles.
Dada la conversación actual, decidís quién debe intervenir ahora o si es momento de finalizar.

Especialistas disponibles:
- "investigador": busca reglas y datos en el manual técnico del ERP. No calcula.
- "analista": hace cálculos (fechas, porcentajes, montos) con los datos del investigador. No busca.
- "FINALIZAR": la información alcanza; un sintetizador redacta la respuesta final.

RÚBRICA DE VALIDACIÓN (revisala antes de elegir FINALIZAR):
1. Cada regla de negocio usada (plazos, porcentajes, roles) la aportó el investigador con su fuente.
2. Cada número calculado lo aportó el analista usando herramientas y a partir de esas reglas.
3. Están respondidas TODAS las partes de la pregunta del usuario.
4. No hay contradicciones entre los aportes ni datos marcados como "no encontrado" o "falta".
Si un criterio falla, devolvé el trabajo al especialista que corresponda con una instrucción concreta
de qué corregir o completar. Si todos se cumplen, elegí FINALIZAR. No pidas trabajo que ya está hecho.

Respondé SOLO con un JSON, sin texto antes ni después:
{"siguiente": "investigador" | "analista" | "FINALIZAR",
 "instruccion": "tarea concreta para el especialista (vacío si FINALIZAR)",
 "motivo": "por qué, en una oración"}"""


class DecisionLLM(BaseModel):
    siguiente: Literal["investigador", "analista", "FINALIZAR"]
    instruccion: str = ""
    motivo: str = Field(default="")


def _pizarra(contribuciones: list[Contribucion]) -> str:
    if not contribuciones:
        return "(todavía no hay contribuciones)"
    bloques = []
    for c in contribuciones:
        herramientas = ", ".join(c["herramientas"]) or "NINGUNA"
        fuentes = f" · fuentes: {', '.join(c['fuentes'])}" if c["fuentes"] else ""
        bloques.append(f"[ronda {c['ronda']}] {c['agente'].upper()} (encargo: {c['instruccion']})\n"
                       f"herramientas usadas: {herramientas}{fuentes}\n{c['resultado']}")
    return "\n\n".join(bloques)


def _parsear(texto: str) -> DecisionLLM:
    match = re.search(r"\{.*\}", texto, re.S)
    if not match:
        raise ValueError("la respuesta no contiene un JSON")
    return DecisionLLM.model_validate(json.loads(match.group(0)))


def _regla_por_defecto(contribuciones: list[Contribucion]) -> tuple[Destino, str]:
    """Decisión determinista si el modelo no responde algo interpretable."""
    agentes = {c["agente"] for c in contribuciones}
    if "investigador" not in agentes:
        return "investigador", "Buscá en el manual las reglas necesarias para responder la pregunta."
    if "analista" not in agentes:
        return "analista", "Hacé los cálculos que pide la pregunta con los hallazgos del investigador."
    return "sintetizador", ""


def crear_nodo(llm: BaseChatModel):
    async def supervisor(state: OrchestratorState) -> dict[str, Any]:
        contribuciones = state.get("contribuciones", [])
        delegaciones = state.get("delegaciones", 0)
        ronda = delegaciones + 1

        # Salvaguarda 1: tope total de delegaciones → cerrar con lo que haya
        if delegaciones >= config.MAX_DELEGACIONES:
            motivo = f"Se alcanzó el tope de {config.MAX_DELEGACIONES} delegaciones: se cierra con lo disponible."
            logger.warning(motivo)
            return {"next_agent": "sintetizador", "instruccion": "",
                    "decisiones": [{"ronda": ronda, "siguiente": "sintetizador", "motivo": motivo, "origen": "regla"}]}

        pedido = (f"Pregunta del usuario:\n{pregunta_original(state)}\n\n"
                  f"Contribuciones hasta ahora:\n{_pizarra(contribuciones)}\n\n"
                  f"Delegaciones usadas: {delegaciones} de {config.MAX_DELEGACIONES}.")
        mensajes = [SystemMessage(PROMPT), HumanMessage(pedido)]
        decision: DecisionLLM | None = None
        for intento in range(2):   # un reintento si el JSON viene mal
            respuesta = await llm.ainvoke(mensajes)
            try:
                decision = _parsear(texto_de(respuesta))
                break
            except (ValueError, ValidationError) as e:
                logger.warning("Respuesta del supervisor no interpretable (intento %d): %s", intento + 1, e)
                mensajes += [respuesta, HumanMessage("Tu respuesta no es un JSON válido con el formato pedido. "
                                                     "Respondé SOLO el JSON.")]

        if decision is None:
            destino, instruccion = _regla_por_defecto(contribuciones)
            motivo, origen = "El modelo no devolvió una decisión válida: se aplicó la regla por defecto.", "regla"
        else:
            destino = "sintetizador" if decision.siguiente == "FINALIZAR" else decision.siguiente
            instruccion, motivo, origen = decision.instruccion, decision.motivo, "llm"

        # Salvaguarda 2: un especialista no puede ser revisado indefinidamente
        aportes = Counter(c["agente"] for c in contribuciones)
        if destino != "sintetizador" and aportes[destino] > config.MAX_REVISIONES:
            motivo = (f"{destino} ya intervino {aportes[destino]} veces (tope de revisiones: {config.MAX_REVISIONES}): "
                      "se cierra con lo disponible.")
            destino, instruccion, origen = "sintetizador", "", "regla"
            logger.warning(motivo)

        logger.info("ronda %d → %s (%s) · %s", ronda, destino, origen, motivo)
        actualizacion: dict[str, Any] = {
            "next_agent": destino, "instruccion": instruccion,
            "decisiones": [{"ronda": ronda, "siguiente": destino, "motivo": motivo, "origen": origen}],
        }
        if destino != "sintetizador":
            actualizacion["delegaciones"] = ronda   # el contador solo avanza cuando se delega
        return actualizacion

    return supervisor


def route_supervisor(state: OrchestratorState) -> Literal["investigador", "analista", "sintetizador"]:
    """Arista condicional: mapea la decisión del supervisor al nombre del nodo del grafo."""
    return state["next_agent"]
