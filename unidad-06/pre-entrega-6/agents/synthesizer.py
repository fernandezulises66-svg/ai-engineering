"""Sintetizador: redacta la respuesta final para el usuario a partir de las contribuciones validadas."""

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.common import diagnostico, pregunta_original, texto_de
from agents.supervisor import _pizarra
from state import OrchestratorState

logger = logging.getLogger("sintetizador")

PROMPT = """Redactás la respuesta final para un usuario del ERP de OLA Muebles, en español rioplatense.

- Usá SOLO la información de las contribuciones de los especialistas; no agregues datos propios.
- Si un mismo especialista aportó varias veces, la versión de la ronda más alta reemplaza a las anteriores.
- Respondé TODAS las partes de la pregunta en una lista numerada (una entrada por cada pregunta del
  usuario), de forma breve, y citá entre corchetes el documento del manual de cada regla (ej.: [03-api-ventas.md]).
- Si una regla cubre el caso (ej.: "más del 15 %" cubre un 18 %), aplicala con seguridad.
- Si algo quedó sin resolver o los aportes se contradicen, decilo claramente en vez de elegir uno al azar.
- Mencioná un supuesto solo si cambia el resultado, en una línea al final."""


def crear_nodo(llm: BaseChatModel):
    async def sintetizador(state: OrchestratorState) -> dict[str, Any]:
        pedido = (f"Pregunta del usuario:\n{pregunta_original(state)}\n\n"
                  f"Contribuciones:\n{_pizarra(state.get('contribuciones', []))}")
        mensaje = await llm.ainvoke([SystemMessage(PROMPT), HumanMessage(pedido)])
        respuesta, diag = texto_de(mensaje).strip(), diagnostico(mensaje)
        logger.info("respuesta final lista (%d caracteres, stop_reason=%s)", len(respuesta), diag["stop_reason"])
        if diag["stop_reason"] == "max_tokens":
            logger.warning("La respuesta final se cortó por max_tokens")
        return {"respuesta_final": respuesta, "task_completed": True, "diagnostico_sintesis": diag,
                "messages": [AIMessage(respuesta, name="sintetizador")]}

    return sintetizador
