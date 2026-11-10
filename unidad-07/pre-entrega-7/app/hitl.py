"""
Human-in-the-loop: pausa obligatoria antes de cualquier acción con efectos secundarios en el ERP.

Flujo dentro del grafo (después de que el sintetizador redactó la respuesta):

    sintetizador → evaluar_riesgo ─┬─(consulta)──────────────────────────────► END
                                   └─(crítica)─► aprobacion_humana ─┬─(sí)─► ejecutar_accion → END
                                                   [interrupt]      └─(no)─────────────────────► END

- `evaluar_riesgo` decide con REGLAS (no con el LLM) si la tarea es crítica: si el usuario pide
  ejecutar algo en el ERP (aplicá, registrá, anulá, emití…). Una compuerta de seguridad no debería
  depender de que el modelo la respete.
- `aprobacion_humana` llama a `interrupt()`: LangGraph guarda el estado en Redis (checkpointer) y
  la ejecución se detiene. El job queda en WAITING_APPROVAL hasta que alguien llame a
  POST /tasks/{id}/approve; ahí el worker reanuda el grafo con `Command(resume=decisión)`.
- `ejecutar_accion` es la única parte con efectos secundarios: escribe en el "ERP" (una lista de Redis).
"""

import logging
import re
import unicodedata
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request, status
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from app.agents.common import pregunta_original
from app.jobs import JobStatus
from app.state import AccionPropuesta, OrchestratorState

logger = logging.getLogger("hitl")

# Verbos de acción en imperativo (voseo y tuteo con acento/clítico) → tipo de acción en el ERP
ACCIONES: dict[str, str] = {
    "aplic": "aplicar_descuento", "registr": "registrar_operacion", "carg": "registrar_operacion",
    "anul": "anular_venta", "emit": "emitir_comprobante", "factur": "emitir_comprobante",
    "aprob": "aprobar_descuento", "confirm": "confirmar_venta", "elimin": "eliminar_registro",
    "borr": "eliminar_registro", "modific": "modificar_registro", "cobr": "registrar_cobro",
}
_IMPERATIVO = re.compile(
    r"\b(aplic|registr|carg|anul|emit|factur|aprob|confirm|elimin|borr|modific|cobr)"
    r"(a|e|i)(lo|la|los|las|le|les)\b", re.IGNORECASE)
_IMPERATIVO_VOSEO = re.compile(
    r"\b(aplic|registr|carg|anul|emit|factur|aprob|confirm|elimin|borr|modific|cobr)(á|é|í)\b", re.IGNORECASE)


def _sin_acentos(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c))


def detectar_criticidad(pregunta: str) -> tuple[bool, str, str]:
    """(es_crítica, motivo, tipo_de_acción). Crítica = el usuario pide EJECUTAR algo en el ERP."""
    match = _IMPERATIVO_VOSEO.search(pregunta)
    if not match:
        # "aplicalo", "registrala", "confirmalo": imperativo con pronombre, escrito sin acento
        match = _IMPERATIVO.search(_sin_acentos(pregunta))
    if not match:
        return False, "Consulta de solo lectura: no pide ejecutar acciones en el ERP.", ""
    raiz = match.group(1).lower()
    tipo = ACCIONES[raiz]
    return True, f"Pide ejecutar una acción con efectos secundarios en el ERP ('{match.group(0)}' → {tipo}).", tipo


# --- Nodos del grafo --------------------------------------------------------------------
async def evaluar_riesgo(state: OrchestratorState) -> dict[str, Any]:
    pregunta = pregunta_original(state)
    critica, motivo, tipo = detectar_criticidad(pregunta)
    logger.info("criticidad: %s · %s", critica, motivo)
    accion: AccionPropuesta | None = None
    if critica:
        accion = {"tipo": tipo, "descripcion": f"{tipo.replace('_', ' ').capitalize()}: {pregunta[:240]}",
                  "detalle": state.get("respuesta_final", "")}
    return {"es_critica": critica, "motivo_criticidad": motivo, "accion_propuesta": accion}


def route_riesgo(state: OrchestratorState) -> Literal["aprobacion_humana", "__end__"]:
    return "aprobacion_humana" if state.get("es_critica") else END


async def aprobacion_humana(state: OrchestratorState) -> dict[str, Any]:
    """Pausa obligatoria. `interrupt()` corta la ejecución acá; al reanudar devuelve la decisión humana.
    No hay efectos secundarios antes del interrupt: al reanudar, el nodo se vuelve a ejecutar desde el inicio."""
    decision = interrupt({
        "tipo": "aprobacion_requerida",
        "motivo": state.get("motivo_criticidad", ""),
        "accion_propuesta": state.get("accion_propuesta"),
        "como_responder": "POST /tasks/{job_id}/approve con {\"aprobado\": true|false, \"revisor\": \"...\", \"comentario\": \"...\"}",
    })
    aprobacion = {"aprobado": bool(decision.get("aprobado")), "revisor": str(decision.get("revisor", "")),
                  "comentario": str(decision.get("comentario", ""))}
    logger.info("decisión humana: %s", aprobacion)
    actualizacion: dict[str, Any] = {"aprobacion": aprobacion}
    if not aprobacion["aprobado"]:
        motivo = f"Acción rechazada por {aprobacion['revisor'] or 'el revisor'}: no se modificó el ERP."
        actualizacion |= {"accion_resultado": motivo, "messages": [AIMessage(motivo, name="hitl")]}
    return actualizacion


def route_aprobacion(state: OrchestratorState) -> Literal["ejecutar_accion", "__end__"]:
    return "ejecutar_accion" if (state.get("aprobacion") or {}).get("aprobado") else END


Registrador = Callable[[str, dict[str, Any], dict[str, Any]], Awaitable[str]]


def crear_nodo_ejecutar(registrar: Registrador):
    async def ejecutar_accion(state: OrchestratorState, config: RunnableConfig) -> dict[str, Any]:
        job_id = config["configurable"]["thread_id"]
        resultado = await registrar(job_id, dict(state["accion_propuesta"]), dict(state["aprobacion"]))
        logger.info(resultado)
        return {"accion_resultado": resultado, "messages": [AIMessage(resultado, name="ejecutar_accion")]}

    return ejecutar_accion


# --- Endpoint de aprobación ----------------------------------------------------------------
router = APIRouter(tags=["human-in-the-loop"])


class Decision(BaseModel):
    aprobado: bool = True
    revisor: str = Field(default="revisor", max_length=80)
    comentario: str = Field(default="", max_length=500)


@router.post("/tasks/{job_id}/approve", status_code=status.HTTP_202_ACCEPTED)
async def aprobar(job_id: str, decision: Decision, request: Request) -> dict[str, Any]:
    """Registra la decisión humana y reencola el job para que el worker reanude el grafo."""
    store = request.app.state.store
    job = await store.obtener(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No existe ese job")
    if job["status"] != JobStatus.WAITING_APPROVAL:
        raise HTTPException(status.HTTP_409_CONFLICT, f"El job no espera aprobación (estado: {job['status']})")
    await store.actualizar(job_id, JobStatus.PENDING, aprobacion=decision.model_dump())
    await store.borrar_campo(job_id, "aprobacion_pendiente")
    await store.encolar(job_id, "resume", decision.model_dump())
    return {"job_id": job_id, "status": JobStatus.PENDING, "decision": decision.model_dump()}
