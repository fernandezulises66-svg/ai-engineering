"""
Pipeline LCEL de extracción de entidades técnicas.

    prompt | model.with_structured_output(ExtraccionTecnica) | validar_salida
    └──────────────────── .with_retry() ──────────────────────────────────┘

- El prompt es un ChatPromptTemplate modular (sin f-strings dentro de la cadena).
- with_structured_output usa tool calling para que Claude devuelva el esquema;
  include_raw=True nos deja ver el mensaje crudo (stop_reason) y el error de parseo.
- validar_salida detecta respuestas cortadas por max_tokens, JSON mal formado o
  campos inválidos y lanza una excepción, que dispara el reintento automático.
"""

import asyncio
import logging
import os
from typing import Any

from dotenv import find_dotenv, load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import ValidationError

from schemas import ExtraccionTecnica

load_dotenv(find_dotenv())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")
# Silenciamos el ruido de las librerías HTTP para que se lean los logs del pipeline
logging.getLogger("httpx").setLevel(logging.WARNING)

MAX_INTENTOS = 3


class SalidaIncompletaError(Exception):
    """El modelo cortó la respuesta (stop_reason = max_tokens) antes de terminar el JSON."""


# --- 1. Modelo (reutiliza la configuración del Módulo 1) ----------------------
def build_model() -> ChatAnthropic:
    params: dict[str, Any] = {
        "model": os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5"),
        "max_tokens": int(os.getenv("ANTHROPIC_MAX_TOKENS", "1024")),
        "max_retries": 2,  # reintentos de red / 429 (capa HTTP)
        "timeout": 30,
    }
    # Claude 5.x rechaza temperature (400). Solo se envía si se configuró a mano.
    if temperature := os.getenv("ANTHROPIC_TEMPERATURE"):
        params["temperature"] = float(temperature)
    return ChatAnthropic(**params)


# --- 2. Prompt modular --------------------------------------------------------
INSTRUCCIONES_FORMATO = """Devolvé la información usando la herramienta de salida estructurada:
- tecnologias: solo las que aparecen en el texto, con su nombre propio. Nunca inventes.
- nivel_de_criticidad: "baja", "media" o "alta" según el impacto descripto.
- resumen_tecnico: una o dos oraciones en español, técnicas y concretas."""

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Sos un ingeniero de software senior que analiza textos técnicos "
            "(logs de error, descripciones de arquitectura, incidentes) y extrae "
            "entidades de forma precisa.\n\n{instrucciones_formato}",
        ),
        ("human", "Texto a analizar:\n<texto>\n{texto}\n</texto>"),
    ]
).partial(instrucciones_formato=INSTRUCCIONES_FORMATO)


# --- 3. Validación de la salida (dispara los reintentos) ----------------------
def validar_salida(resultado: dict[str, Any]) -> ExtraccionTecnica:
    raw = resultado.get("raw")
    stop_reason = (getattr(raw, "response_metadata", None) or {}).get("stop_reason")

    if stop_reason == "max_tokens":
        logger.warning("Respuesta cortada por max_tokens: el objeto está incompleto → reintento")
        raise SalidaIncompletaError("la respuesta se cortó por max_tokens")

    if (error := resultado.get("parsing_error")) is not None:
        logger.warning("Salida inválida (%s): %s → reintento", type(error).__name__, error)
        raise error

    parsed = resultado.get("parsed")
    if parsed is None:
        logger.warning("El modelo no devolvió la estructura esperada → reintento")
        raise OutputParserException("el modelo no devolvió la salida estructurada")

    logger.info("Validación OK (stop_reason=%s)", stop_reason)
    return parsed


# --- 4. Cadena LCEL con reintentos --------------------------------------------
def build_chain(llm: BaseChatModel | None = None) -> Runnable:
    llm = llm or build_model()
    structured_llm = llm.with_structured_output(ExtraccionTecnica, include_raw=True)

    log_intento = RunnableLambda(_log_intento)
    chain = prompt | log_intento | structured_llm | RunnableLambda(validar_salida)

    return chain.with_retry(
        retry_if_exception_type=(OutputParserException, ValidationError, SalidaIncompletaError),
        stop_after_attempt=MAX_INTENTOS,
        wait_exponential_jitter=True,
    )


def _log_intento(prompt_value: Any) -> Any:
    logger.info("Llamando al modelo...")
    return prompt_value


_chain: Runnable | None = None


def get_chain() -> Runnable:
    """Crea la cadena una sola vez (lazy) para no exigir la API key al importar el módulo."""
    global _chain
    if _chain is None:
        _chain = build_chain()
    return _chain


# --- 5. Punto de entrada asíncrono --------------------------------------------
async def process_text(text: str, chain: Runnable | None = None) -> ExtraccionTecnica:
    """Procesa un texto técnico y devuelve un objeto ExtraccionTecnica validado."""
    if not text or not text.strip():
        raise ValueError("el texto de entrada está vacío")

    chain = chain or get_chain()
    logger.info("Procesando texto (%d caracteres)", len(text))
    try:
        resultado = await chain.ainvoke({"texto": text.strip()})
    except (OutputParserException, ValidationError, SalidaIncompletaError) as exc:
        logger.error("No se obtuvo una salida válida tras %d intentos: %s", MAX_INTENTOS, exc)
        raise
    logger.info("Extracción completada: %d tecnologías, criticidad %s",
                len(resultado.tecnologias), resultado.nivel_de_criticidad.value)
    return resultado


if __name__ == "__main__":
    ejemplo = (
        "ERROR 2026-10-14 03:12:44 api-gateway: psycopg2.OperationalError: "
        "too many connections for role 'app'. FastAPI devuelve 503 en /checkout; "
        "la caché de Redis no alcanza a cubrir las lecturas."
    )
    print(asyncio.run(process_text(ejemplo)).model_dump_json(indent=2))
