"""
API REST del orquestador multi-agente.

    POST /tasks                 encola la pregunta y devuelve un job_id (202, no espera al agente)
    GET  /tasks/{job_id}        estado del job (polling): PENDING, RUNNING, WAITING_APPROVAL, DONE o FAILED
    POST /tasks/{job_id}/approve  decisión humana para un job pausado (ver hitl.py)
    GET  /health                Redis responde

Ningún endpoint llama al LLM ni hace I/O bloqueante: solo operaciones async contra Redis. El trabajo
pesado lo hace el worker en segundo plano.

    uvicorn app.main:app --reload
"""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field
from redis.asyncio import Redis, from_url

from app import config, hitl
from app.graph import build_graph, redis_checkpointer
from app.jobs import JobStatus, JobStore
from app.observability import init_observability
from app.worker import Worker

logger = logging.getLogger("api")


class TaskIn(BaseModel):
    pregunta: str = Field(min_length=5, max_length=2000, examples=[
        "Hoy es 10/11/2026. Un presupuesto del 28/10/2026, ¿sigue vigente?"])


class TaskOut(BaseModel):
    job_id: str
    status: JobStatus
    links: dict[str, str]


def create_app(redis_factory: Callable[[], Redis] | None = None,
               graph_factory: Callable[..., Any] | None = None) -> FastAPI:
    """Fábrica de la app. Los tests inyectan un Redis falso y un grafo con modelos guionados."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        init_observability()
        async with AsyncExitStack() as stack:
            redis = redis_factory() if redis_factory else from_url(config.REDIS_URL)
            stack.push_async_callback(redis.aclose)
            store = JobStore(redis)
            if graph_factory:
                graph = graph_factory(store)
            else:
                saver = await stack.enter_async_context(redis_checkpointer())
                graph = build_graph(saver, store.registrar_accion_erp)
            app.state.store, app.state.graph = store, graph
            if config.WORKER_EMBEDDED:
                worker = Worker(store, graph)
                worker.start()
                stack.push_async_callback(worker.stop)
            logger.info("API lista (worker embebido: %s)", config.WORKER_EMBEDDED)
            yield

    app = FastAPI(title="Orquestador multi-agente · OLA Muebles ERP", version="0.7.0", lifespan=lifespan)
    app.include_router(hitl.router)

    @app.post("/tasks", status_code=status.HTTP_202_ACCEPTED, response_model=TaskOut)
    async def crear_tarea(tarea: TaskIn, request: Request) -> TaskOut:
        """Encola la tarea y responde enseguida con el job_id. El worker la procesa en segundo plano."""
        store: JobStore = request.app.state.store
        job = await store.crear(tarea.pregunta)
        await store.encolar(job["job_id"], "start")
        return TaskOut(job_id=job["job_id"], status=JobStatus.PENDING,
                       links={"estado": f"/tasks/{job['job_id']}", "aprobar": f"/tasks/{job['job_id']}/approve"})

    @app.get("/tasks/{job_id}")
    async def estado_tarea(job_id: str, request: Request) -> dict[str, Any]:
        job = await request.app.state.store.obtener(job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "No existe ese job (o ya expiró)")
        return job

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        ok = await request.app.state.store.redis.ping()
        return {"status": "ok" if ok else "degradado", "redis": bool(ok), "worker_embebido": config.WORKER_EMBEDDED}

    return app


app = create_app()
