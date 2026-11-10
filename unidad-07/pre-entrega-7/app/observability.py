"""
Observabilidad con Arize Phoenix (OpenTelemetry + OpenInference).

- `init_observability()` registra el exportador OTLP hacia Phoenix e instrumenta LangChain/LangGraph:
  cada nodo del grafo, cada llamada al LLM (con tokens de entrada/salida, de donde Phoenix calcula
  el costo) y cada herramienta quedan como spans de una misma traza.
- `span_job()` abre la traza raíz de cada ejecución de un job y la agrupa por sesión (= job_id), así
  el arranque y la reanudación tras la aprobación humana se ven juntos en Phoenix.
- `@observar(...)` decora funciones propias que no son Runnables de LangChain (ej.: la búsqueda en
  Pinecone) para que también aparezcan en la traza.
"""

import functools
import json
import logging
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from typing import Any, ParamSpec, TypeVar

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from app import config

logger = logging.getLogger("observabilidad")

P = ParamSpec("P")
R = TypeVar("R")
_tracer: Any = None


def init_observability() -> bool:
    """Conecta con Phoenix e instrumenta LangChain. Devuelve False si está desactivado o falla."""
    global _tracer
    if not config.PHOENIX_ENABLED:
        logger.info("Observabilidad desactivada (PHOENIX_ENABLED=false)")
        return False
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from phoenix.otel import register

        provider = register(project_name=config.PHOENIX_PROJECT, endpoint=config.PHOENIX_ENDPOINT,
                            batch=True, verbose=False)
        LangChainInstrumentor().instrument(tracer_provider=provider)
        _tracer = provider.get_tracer("ola-erp-api")
        logger.info("Trazas → Phoenix %s (proyecto '%s')", config.PHOENIX_ENDPOINT, config.PHOENIX_PROJECT)
        return True
    except Exception as e:  # la API sigue funcionando aunque Phoenix no esté disponible
        logger.warning("No se pudo inicializar Phoenix: %s", e)
        return False


def tracer() -> Any:
    return _tracer or trace.get_tracer("ola-erp-api")   # sin Phoenix: tracer no-op de OpenTelemetry


def _json(valor: Any) -> str:
    return json.dumps(valor, ensure_ascii=False, default=str)[:4000]


@contextmanager
def span_job(job_id: str, tipo: str, entrada: Any) -> Iterator[Any]:
    """Traza raíz de una ejecución del job: kind CHAIN, sesión = job_id, metadata con el tipo."""
    from openinference.instrumentation import using_metadata, using_session

    with using_session(job_id), using_metadata({"job_id": job_id, "tipo": tipo}):
        with tracer().start_as_current_span(f"job.{tipo}", attributes={
                "openinference.span.kind": "CHAIN", "input.value": _json(entrada),
                "input.mime_type": "application/json", "session.id": job_id}) as span:
            yield span


def observar(nombre: str, kind: str = "CHAIN") -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Decorador para funciones async propias: crea un span con su entrada, salida y errores."""
    def decorador(fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @functools.wraps(fn)
        async def envuelta(*args: P.args, **kwargs: P.kwargs) -> R:
            entrada = kwargs or (args[1:] if args and hasattr(args[0], "__dict__") else args)
            with tracer().start_as_current_span(nombre, attributes={
                    "openinference.span.kind": kind, "input.value": _json(entrada)}) as span:
                try:
                    resultado = await fn(*args, **kwargs)
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
                span.set_attribute("output.value", _json(resultado))
                return resultado
        return envuelta
    return decorador
