"""
Tests sin API key: reemplazan el LLM por un modelo falso que devuelve respuestas
predefinidas, para verificar la validación y los reintentos de la cadena.

Ejecutar desde la carpeta pre-entrega-2:
    pytest -v
"""

import asyncio

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from chain import MAX_INTENTOS, build_chain, process_text
from schemas import ExtraccionTecnica, NivelCriticidad

VALIDO = ExtraccionTecnica(
    tecnologias=["FastAPI", "Redis", "PostgreSQL"],
    nivel_de_criticidad="alta",
    resumen_tecnico="API con caché en Redis; se agotan las conexiones a PostgreSQL.",
)


def _validation_error() -> ValidationError:
    try:
        ExtraccionTecnica.model_validate({"tecnologias": [], "nivel_de_criticidad": "urgente"})
    except ValidationError as exc:
        return exc
    raise AssertionError("debería haber fallado")


def ok(stop_reason: str = "tool_use") -> dict:
    raw = AIMessage(content="", response_metadata={"stop_reason": stop_reason})
    return {"raw": raw, "parsed": VALIDO, "parsing_error": None}


def json_invalido() -> dict:
    raw = AIMessage(content="", response_metadata={"stop_reason": "tool_use"})
    return {"raw": raw, "parsed": None, "parsing_error": _validation_error()}


def cortado() -> dict:
    raw = AIMessage(content="", response_metadata={"stop_reason": "max_tokens"})
    return {"raw": raw, "parsed": None, "parsing_error": None}


class FakeLLM:
    """Imita model.with_structured_output(..., include_raw=True) con respuestas en cola."""

    def __init__(self, *respuestas: dict) -> None:
        self.respuestas = list(respuestas)
        self.llamadas = 0

    def with_structured_output(self, schema, include_raw=False, **kwargs):
        assert schema is ExtraccionTecnica and include_raw

        def responder(_prompt):
            self.llamadas += 1
            return self.respuestas.pop(0)

        return RunnableLambda(responder)


def correr(llm: FakeLLM, texto: str = "Log con FastAPI, Redis y PostgreSQL caídos"):
    return asyncio.run(process_text(texto, chain=build_chain(llm)))


# --- Cadena y reintentos -----------------------------------------------------

def test_salida_valida_al_primer_intento():
    llm = FakeLLM(ok())
    assert correr(llm) == VALIDO
    assert llm.llamadas == 1


def test_reintenta_ante_json_invalido():
    llm = FakeLLM(json_invalido(), ok())
    assert correr(llm) == VALIDO
    assert llm.llamadas == 2


def test_reintenta_si_la_respuesta_se_corta_por_max_tokens():
    llm = FakeLLM(cortado(), ok())
    assert correr(llm) == VALIDO
    assert llm.llamadas == 2


def test_falla_despues_de_agotar_los_intentos():
    llm = FakeLLM(*[json_invalido() for _ in range(MAX_INTENTOS)])
    with pytest.raises(ValidationError):
        correr(llm)
    assert llm.llamadas == MAX_INTENTOS


def test_texto_vacio_no_llama_al_modelo():
    llm = FakeLLM(ok())
    with pytest.raises(ValueError):
        correr(llm, texto="   ")
    assert llm.llamadas == 0


# --- Esquema Pydantic --------------------------------------------------------

def test_esquema_limpia_y_deduplica_tecnologias():
    r = ExtraccionTecnica(
        tecnologias=[" Redis ", "redis", "FastAPI", ""],
        nivel_de_criticidad="media",
        resumen_tecnico="Resumen técnico de prueba con longitud suficiente.",
    )
    assert r.tecnologias == ["Redis", "FastAPI"]
    assert r.nivel_de_criticidad is NivelCriticidad.MEDIA


@pytest.mark.parametrize(
    "datos",
    [
        {"tecnologias": [], "nivel_de_criticidad": "alta", "resumen_tecnico": "x" * 30},
        {"tecnologias": ["  "], "nivel_de_criticidad": "alta", "resumen_tecnico": "x" * 30},
        {"tecnologias": ["Redis"], "nivel_de_criticidad": "urgente", "resumen_tecnico": "x" * 30},
        {"tecnologias": ["Redis"], "nivel_de_criticidad": "alta", "resumen_tecnico": "corto"},
        {"tecnologias": ["Redis"], "nivel_de_criticidad": "alta"},
    ],
    ids=["lista-vacia", "solo-espacios", "enum-invalido", "resumen-corto", "falta-campo"],
)
def test_esquema_rechaza_datos_invalidos(datos):
    with pytest.raises(ValidationError):
        ExtraccionTecnica.model_validate(datos)
