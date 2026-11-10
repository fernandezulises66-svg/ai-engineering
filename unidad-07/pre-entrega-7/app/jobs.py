"""
Estado de los jobs, cola de trabajo y "ERP" simulado, todo en Redis.

Claves:
- pe7:job:{id}           hash con el estado del job (status, pregunta, resultado, error, timestamps…)
- pe7:job:{id}:eventos   lista con el historial de transiciones (auditoría)
- pe7:queue              lista usada como cola (RPUSH para encolar, BLPOP para consumir)
- erp:acciones           lista con las acciones que el sistema ejecutó en el ERP tras la aprobación humana

Los checkpoints de LangGraph van aparte, con el AsyncRedisSaver (ver graph.py).
"""

import json
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from redis.asyncio import Redis

from app import config

PREFIJO = "pe7"
COLA = f"{PREFIJO}:queue"
ERP_ACCIONES = "erp:acciones"


class JobStatus(StrEnum):
    PENDING = "PENDING"                    # encolado, esperando un worker
    RUNNING = "RUNNING"                    # el grafo está corriendo
    WAITING_APPROVAL = "WAITING_APPROVAL"  # pausado por el nodo Human-in-the-loop
    DONE = "DONE"
    FAILED = "FAILED"


def _ahora() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class JobStore:
    def __init__(self, redis: Redis) -> None:
        self.redis = redis

    @staticmethod
    def _clave(job_id: str) -> str:
        return f"{PREFIJO}:job:{job_id}"

    async def crear(self, pregunta: str) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        ahora = _ahora()
        job = {"job_id": job_id, "status": JobStatus.PENDING, "pregunta": pregunta,
               "created_at": ahora, "updated_at": ahora}
        clave = self._clave(job_id)
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.hset(clave, mapping=job)
            pipe.expire(clave, config.JOB_TTL_SEGUNDOS)
            pipe.rpush(f"{clave}:eventos", json.dumps({"status": JobStatus.PENDING, "at": ahora}))
            pipe.expire(f"{clave}:eventos", config.JOB_TTL_SEGUNDOS)
            await pipe.execute()
        return job

    async def obtener(self, job_id: str) -> dict[str, Any] | None:
        datos = await self.redis.hgetall(self._clave(job_id))
        if not datos:
            return None
        job: dict[str, Any] = {(k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
                               for k, v in datos.items()}
        for campo in ("resultado", "aprobacion_pendiente", "aprobacion"):
            if campo in job:
                job[campo] = json.loads(job[campo])
        eventos = await self.redis.lrange(f"{self._clave(job_id)}:eventos", 0, -1)
        job["eventos"] = [json.loads(e) for e in eventos]
        return job

    async def actualizar(self, job_id: str, status: JobStatus | None = None, **campos: Any) -> None:
        """Actualiza campos del hash (HSET es atómico por campo). Los dict/list se guardan como JSON."""
        clave = self._clave(job_id)
        valores = {k: (json.dumps(v, ensure_ascii=False, default=str) if isinstance(v, (dict, list)) else str(v))
                   for k, v in campos.items() if v is not None}
        ahora = _ahora()
        valores["updated_at"] = ahora
        async with self.redis.pipeline(transaction=True) as pipe:
            if status:
                valores["status"] = status
                pipe.rpush(f"{clave}:eventos", json.dumps({"status": status, "at": ahora}))
            pipe.hset(clave, mapping=valores)
            await pipe.execute()

    async def borrar_campo(self, job_id: str, campo: str) -> None:
        await self.redis.hdel(self._clave(job_id), campo)

    # --- Cola --------------------------------------------------------------------------
    async def encolar(self, job_id: str, tipo: str = "start", payload: dict[str, Any] | None = None) -> None:
        await self.redis.rpush(COLA, json.dumps({"job_id": job_id, "tipo": tipo, "payload": payload or {}}))

    async def desencolar(self, timeout: float = 1.0) -> dict[str, Any] | None:
        item = await self.redis.blpop([COLA], timeout=timeout)
        return json.loads(item[1]) if item else None

    # --- "ERP" simulado: la herramienta con efectos secundarios ---------------------------
    async def registrar_accion_erp(self, job_id: str, accion: dict[str, Any], aprobacion: dict[str, Any]) -> str:
        registro = {"job_id": job_id, "accion": accion, "aprobado_por": aprobacion.get("revisor", "?"),
                    "comentario": aprobacion.get("comentario", ""), "registrado_at": _ahora()}
        posicion = await self.redis.rpush(ERP_ACCIONES, json.dumps(registro, ensure_ascii=False))
        return f"Acción '{accion['tipo']}' registrada en el ERP (registro #{posicion}), aprobada por {registro['aprobado_por']}."
