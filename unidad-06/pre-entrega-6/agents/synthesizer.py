"""Sintetizador: redacta la respuesta final para el usuario a partir de las contribuciones validadas."""

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.common import pregunta_original, texto_de
from agents.supervisor import _pizarra
from state import OrchestratorState

logger = logging.getLogger("sintetizador")

PROMPT = """Redactás la respuesta final para un usuario del ERP de OLA Muebles, en español rioplatense.

- Usá SOLO la información de las contribuciones de los especialistas; no agregues datos propios.
- Si un mismo especialista aportó varias veces, la versión de la ronda más alta reemplaza a las anteriores.
- Respondé cada parte de la pregunta, de forma breve, y citá entre corchetes el documento del manual
  de cada regla (ej.: [03-api-ventas.md]).
- Si algo quedó sin resolver o los aportes se contradicen, decilo claramente en vez de elegir uno al azar."""


def crear_nodo(llm: BaseChatModel):
    async def sintetizador(state: OrchestratorState) -> dict[str, Any]:
        pedido = (f"Pregunta del usuario:\n{pregunta_original(state)}\n\n"
                  f"Contribuciones:\n{_pizarra(state.get('contribuciones', []))}")
        respuesta = texto_de(await llm.ainvoke([SystemMessage(PROMPT), HumanMessage(pedido)])).strip()
        logger.info("respuesta final lista (%d caracteres)", len(respuesta))
        return {"respuesta_final": respuesta, "task_completed": True,
                "messages": [AIMessage(respuesta, name="sintetizador")]}

    return sintetizador
