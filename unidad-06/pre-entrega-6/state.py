"""
Estado compartido del orquestador.

Además del historial de mensajes (heredado de MessagesState), el estado guarda:
- `contribuciones`: registro de QUÉ agente aportó QUÉ información, en qué ronda y por
  encargo de qué instrucción. Es la "pizarra" común: el supervisor la lee para decidir,
  el analista lee lo que aportó el investigador y el sintetizador arma la respuesta con ella.
- Campos de control del flujo: a quién le toca, qué instrucción lleva, cuántas veces se
  delegó y si la tarea quedó completa.
"""

import operator
from typing import Annotated, Literal, TypedDict

from langgraph.graph import MessagesState

Especialista = Literal["investigador", "analista"]
Destino = Literal["investigador", "analista", "sintetizador"]


class Contribucion(TypedDict):
    agente: Especialista
    ronda: int                  # número de delegación del supervisor (1, 2, 3...)
    instruccion: str            # qué le pidió el supervisor
    resultado: str              # qué devolvió el especialista
    herramientas: list[str]     # herramientas que usó, en orden (ej.: ["buscar_manual", "buscar_manual"])
    fuentes: list[str]          # documentos del manual citados (solo el investigador)


class Decision(TypedDict):
    ronda: int
    siguiente: Destino
    motivo: str
    origen: Literal["llm", "regla"]   # "regla": la decidió una salvaguarda, no el modelo


class OrchestratorState(MessagesState):
    """`messages` (con el reducer add_messages) guarda la conversación con el usuario;
    las listas con `operator.add` acumulan: cada nodo devuelve solo lo nuevo."""

    contribuciones: Annotated[list[Contribucion], operator.add]
    decisiones: Annotated[list[Decision], operator.add]
    next_agent: Destino
    instruccion: str            # tarea concreta para el especialista que sigue
    delegaciones: int           # contador de pasos (condición de parada)
    task_completed: bool
    respuesta_final: str
