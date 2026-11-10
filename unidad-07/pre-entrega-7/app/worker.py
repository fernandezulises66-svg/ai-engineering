"""
Worker: consume la cola de Redis y corre el grafo en segundo plano.

Cada mensaje es {"job_id", "tipo": "start" | "resume", "payload"}:
- start:  ejecuta el grafo desde el principio con la pregunta del job.
- resume: reanuda un job pausado por el Human-in-the-loop con `Command(resume=decisión)`.

Estados que escribe en Redis: RUNNING → DONE | WAITING_APPROVAL | FAILED.
Cualquier excepción se captura y deja el job en FAILED con el error, para que el cliente que
hace polling nunca se quede esperando para siempre.

Por defecto corre dentro del proceso de la API (WORKER_EMBEDDED=true), con N tareas asyncio en
paralelo (WORKER_CONCURRENCY). También se puede correr aparte, en uno o más procesos:

    python -m app.worker
"""

import asyncio
import logging
import time
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from app import config
from app.graph import estado_inicial, run_config
from app.jobs import JobStatus, JobStore
from app.observability import span_job

logger = logging.getLogger("worker")


def _resumen(estado: dict[str, Any]) -> dict[str, Any]:
    """Lo que se guarda como resultado del job (sin los mensajes internos)."""
    return {
        "respuesta_final": estado.get("respuesta_final", ""),
        "recorrido": [d["siguiente"] for d in estado.get("decisiones", [])],
        "delegaciones": estado.get("delegaciones", 0),
        "contribuciones": [{"agente": c["agente"], "ronda": c["ronda"], "herramientas": c["herramientas"],
                            "fuentes": c["fuentes"]} for c in estado.get("contribuciones", [])],
        "es_critica": estado.get("es_critica", False),
        "motivo_criticidad": estado.get("motivo_criticidad", ""),
        "accion_propuesta": estado.get("accion_propuesta"),
        "aprobacion": estado.get("aprobacion"),
        "accion_resultado": estado.get("accion_resultado", ""),
    }


class Worker:
    def __init__(self, store: JobStore, graph: CompiledStateGraph, concurrency: int = config.WORKER_CONCURRENCY) -> None:
        self.store, self.graph, self.concurrency = store, graph, concurrency
        self._tareas: list[asyncio.Task] = []
        self._activo = False

    def start(self) -> None:
        self._activo = True
        self._tareas = [asyncio.create_task(self._loop(i), name=f"worker-{i}") for i in range(self.concurrency)]
        logger.info("Worker iniciado con %d tareas concurrentes", self.concurrency)

    async def stop(self) -> None:
        self._activo = False
        for t in self._tareas:
            t.cancel()
        await asyncio.gather(*self._tareas, return_exceptions=True)

    async def _loop(self, n: int) -> None:
        while self._activo:
            try:
                mensaje = await self.store.desencolar(timeout=1)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("worker-%d: error leyendo la cola; reintento en 1 s", n)
                await asyncio.sleep(1)
                continue
            if mensaje:
                await self.procesar(mensaje)

    async def procesar(self, mensaje: dict[str, Any]) -> None:
        job_id, tipo = mensaje["job_id"], mensaje.get("tipo", "start")
        inicio = time.perf_counter()
        job = await self.store.obtener(job_id)
        if job is None:
            logger.warning("Job %s no existe (¿expiró?): se descarta", job_id)
            return
        entrada: Any = estado_inicial(job["pregunta"]) if tipo == "start" else Command(resume=mensaje.get("payload", {}))
        with span_job(job_id, tipo, {"pregunta": job["pregunta"], "payload": mensaje.get("payload")}) as span:
            try:
                await self.store.actualizar(job_id, JobStatus.RUNNING)
                estado = await self.graph.ainvoke(entrada, run_config(job_id, tipo))
                pausas = estado.get("__interrupt__") or []
                if pausas:
                    await self.store.actualizar(job_id, JobStatus.WAITING_APPROVAL,
                                                aprobacion_pendiente=pausas[0].value, resultado=_resumen(estado))
                    span.set_attribute("output.value", "WAITING_APPROVAL")
                    logger.info("job %s pausado esperando aprobación humana", job_id)
                else:
                    await self.store.actualizar(job_id, JobStatus.DONE, resultado=_resumen(estado),
                                                duracion_ultima_ejecucion_s=round(time.perf_counter() - inicio, 2))
                    span.set_attribute("output.value", estado.get("respuesta_final", "")[:4000])
                    logger.info("job %s terminado en %.1f s", job_id, time.perf_counter() - inicio)
            except asyncio.CancelledError:
                await self.store.actualizar(job_id, JobStatus.FAILED, error="Worker detenido durante la ejecución")
                raise
            except Exception as e:   # sin esto, el cliente haría polling para siempre
                logger.exception("job %s falló", job_id)
                span.record_exception(e)
                await self.store.actualizar(job_id, JobStatus.FAILED, error=f"{type(e).__name__}: {e}"[:1000])


async def main() -> None:
    """Worker como proceso independiente (se puede escalar con varios procesos)."""
    from redis.asyncio import from_url

    from app.graph import build_graph, redis_checkpointer
    from app.observability import init_observability

    init_observability()
    redis = from_url(config.REDIS_URL)
    store = JobStore(redis)
    async with redis_checkpointer() as saver:
        worker = Worker(store, build_graph(saver, store.registrar_accion_erp))
        worker.start()
        try:
            await asyncio.Event().wait()
        finally:
            await worker.stop()
            await redis.aclose()


if __name__ == "__main__":
    asyncio.run(main())
