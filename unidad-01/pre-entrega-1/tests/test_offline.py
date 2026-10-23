"""
Tests sin API keys ni red: usan los SDKs REALES de OpenAI y Anthropic, pero con
un transporte HTTP simulado (httpx.MockTransport) que devuelve respuestas,
streams (SSE) y errores (429, 401, caída de red) predefinidos.

    pytest -v
"""

import asyncio
import json

import httpx
import httpx2  # el SDK de Anthropic usa httpx2; el de OpenAI, httpx
import pytest
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic import ValidationError

from clients import AnthropicClient, OpenAIClient
from manager import AsyncLLMManager
from schemas import ChatMessage, ModelConfig

PREGUNTA = [ChatMessage(role="system", content="Sé breve."), ChatMessage(role="user", content="¿Qué es la entropía?")]


async def no_sleep(_):  # evita esperar el backoff real en los tests
    return None


# --- Respuestas simuladas de cada API --------------------------------------------
def openai_json():
    return {"id": "c1", "object": "chat.completion", "created": 1, "model": "gpt-6-luna",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "La entropía mide el desorden."}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19}}


def openai_sse(tokens):
    lines = []
    for t in tokens:
        chunk = {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "gpt-6-luna",
                 "choices": [{"index": 0, "delta": {"content": t}, "finish_reason": None}]}
        lines.append(f"data: {json.dumps(chunk)}\n\n")
    lines.append("data: [DONE]\n\n")
    return "".join(lines)


def anthropic_json():
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
            "content": [{"type": "text", "text": "La entropía mide el desorden."}],
            "stop_reason": "end_turn", "stop_sequence": None, "usage": {"input_tokens": 12, "output_tokens": 7}}


def anthropic_sse(tokens):
    def ev(name, data):
        return f"event: {name}\ndata: {json.dumps(data)}\n\n"
    out = [ev("message_start", {"type": "message_start", "message": {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-5-5", "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 12, "output_tokens": 1}}}),
        ev("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})]
    out += [ev("content_block_delta", {"type": "content_block_delta", "index": 0,
                                       "delta": {"type": "text_delta", "text": t}}) for t in tokens]
    out += [ev("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ev("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                                 "usage": {"output_tokens": 7}}),
            ev("message_stop", {"type": "message_stop"})]
    return "".join(out)


ERROR_BODY = {"error": {"type": "rate_limit_error", "message": "Too many requests"}}


def resp(status, json=None, headers=None):
    return ("json", status, json, headers or {})


def sse(text):
    return ("sse", 200, text, {"content-type": "text/event-stream"})


CONNECT_ERROR = ("connect_error",)


class FakeAPI:
    """Responde en orden la lista de respuestas configurada y cuenta las llamadas.

    Las respuestas se construyen con la librería HTTP del SDK que llama (httpx o httpx2).
    """

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0
        self.last_body = None
        self.lib = httpx

    def __call__(self, request):
        self.calls += 1
        self.last_body = json.loads(request.content)
        spec = self.responses.pop(0)
        if spec[0] == "connect_error":
            raise self.lib.ConnectError("sin internet", request=request)
        kind, status, payload, headers = spec
        if kind == "sse":
            return self.lib.Response(status, text=payload, headers=headers)
        return self.lib.Response(status, json=payload, headers=headers)


def manager(provider, api: FakeAPI, **cfg):
    config = ModelConfig(provider=provider, model="gpt-6-luna" if provider == "openai" else "claude-haiku-5-5", **cfg)
    if provider == "openai":
        api.lib = httpx
        http = httpx.AsyncClient(transport=httpx.MockTransport(api))
        client = OpenAIClient(config, AsyncOpenAI(api_key="test", http_client=http, max_retries=0))
    else:
        api.lib = httpx2
        http = httpx2.AsyncClient(transport=httpx2.MockTransport(api))
        client = AnthropicClient(config, AsyncAnthropic(api_key="test", http_client=http, max_retries=0))
    return AsyncLLMManager(config, client=client, sleep=no_sleep)


async def juntar(stream):
    return [c async for c in stream]


# --- Modo normal --------------------------------------------------------------------
@pytest.mark.parametrize("provider,body", [("openai", openai_json()), ("anthropic", anthropic_json())])
def test_generate_devuelve_model_response(provider, body):
    api = FakeAPI(resp(200, json=body))
    r = asyncio.run(manager(provider, api).generate(PREGUNTA))
    assert r.ok and r.content == "La entropía mide el desorden."
    assert r.usage.input_tokens == 12 and r.usage.output_tokens == 7


def test_anthropic_recibe_el_system_aparte():
    api = FakeAPI(resp(200, json=anthropic_json()))
    asyncio.run(manager("anthropic", api).generate(PREGUNTA))
    assert api.last_body["system"] == "Sé breve."
    assert [m["role"] for m in api.last_body["messages"]] == ["user"]


def test_temperature_solo_se_envia_si_se_configura():
    api = FakeAPI(resp(200, json=openai_json()), resp(200, json=openai_json()))
    asyncio.run(manager("openai", api).generate("hola"))
    assert "temperature" not in api.last_body
    api2 = FakeAPI(resp(200, json=openai_json()))
    asyncio.run(manager("openai", api2, temperature=0.3).generate("hola"))
    assert api2.last_body["temperature"] == 0.3


# --- Streaming -------------------------------------------------------------------------
@pytest.mark.parametrize("provider,body", [("openai", openai_sse(["La ", "entropía ", "mide..."])),
                                           ("anthropic", anthropic_sse(["La ", "entropía ", "mide..."]))])
def test_stream_hace_yield_de_cada_fragmento(provider, body):
    api = FakeAPI(sse(body))
    chunks = asyncio.run(juntar(manager(provider, api).stream(PREGUNTA)))
    assert [c.delta for c in chunks if c.delta] == ["La ", "entropía ", "mide..."]
    assert chunks[-1].done and chunks[-1].error is None
    assert api.last_body["stream"] is True


def test_stream_reintenta_si_falla_antes_del_primer_token():
    api = FakeAPI(resp(429, json=ERROR_BODY), sse(anthropic_sse(["ok"])))
    chunks = asyncio.run(juntar(manager("anthropic", api).stream("hola")))
    assert [c.delta for c in chunks if c.delta] == ["ok"] and api.calls == 2


def test_stream_devuelve_error_controlado():
    api = FakeAPI(*[resp(429, json=ERROR_BODY)] * 3)
    chunks = asyncio.run(juntar(manager("openai", api).stream("hola")))
    assert chunks[-1].done and chunks[-1].error.type == "rate_limit" and chunks[-1].error.attempts == 3


# --- Errores y reintentos -------------------------------------------------------------
@pytest.mark.parametrize("provider,body", [("openai", openai_json()), ("anthropic", anthropic_json())])
def test_rate_limit_reintenta_y_se_recupera(provider, body):
    api = FakeAPI(resp(429, json=ERROR_BODY), resp(200, json=body))
    r = asyncio.run(manager(provider, api).generate("hola"))
    assert r.ok and api.calls == 2


def test_rate_limit_persistente_devuelve_error_sin_crash():
    api = FakeAPI(*[resp(429, json=ERROR_BODY)] * 3)
    r = asyncio.run(manager("anthropic", api, max_retries=2).generate("hola"))
    assert not r.ok and r.error.type == "rate_limit" and r.error.status_code == 429
    assert r.error.attempts == 3 and api.calls == 3


def test_caida_de_red_se_clasifica_como_connection():
    api = FakeAPI(*[CONNECT_ERROR] * 3)
    r = asyncio.run(manager("openai", api).generate("hola"))
    assert r.error.type == "connection" and r.error.retryable


def test_api_key_invalida_no_se_reintenta():
    api = FakeAPI(resp(401, json={"error": {"type": "authentication_error", "message": "invalid x-api-key"}}))
    r = asyncio.run(manager("anthropic", api).generate("hola"))
    assert r.error.type == "authentication" and not r.error.retryable and api.calls == 1


def test_respeta_el_header_retry_after():
    esperas = []

    async def anotar(s):
        esperas.append(s)

    api = FakeAPI(resp(429, json=ERROR_BODY, headers={"retry-after": "7"}),
                  resp(200, json=openai_json()))
    m = manager("openai", api)
    m._sleep = anotar
    asyncio.run(m.generate("hola"))
    assert esperas == [7.0]


# --- Factory y esquemas -------------------------------------------------------------------
@pytest.mark.parametrize("provider,cls", [("openai", OpenAIClient), ("anthropic", AnthropicClient)])
def test_factory_elige_el_cliente_segun_llm_provider(monkeypatch, provider, cls):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    assert isinstance(AsyncLLMManager.from_env().client, cls)


def test_proveedor_invalido(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    with pytest.raises(ValueError, match="LLM_PROVIDER inválido"):
        AsyncLLMManager.from_env()


@pytest.mark.parametrize("cambio,mensaje", [
    ({"temperature": 2.5}, "La temperatura debe estar entre 0 y 2"),
    ({"temperature": "alta"}, "La temperatura debe ser un número"),
    ({"max_tokens": 5}, "max_tokens debe ser al menos 16"),
    ({"max_tokens": 99999}, "max_tokens no puede superar 32000"),
])
def test_config_invalida_con_mensaje_claro(cambio, mensaje):
    with pytest.raises(ValidationError, match=mensaje):
        ModelConfig(provider="openai", model="gpt-6-luna", **cambio)


def test_mensaje_vacio_es_rechazado():
    with pytest.raises(ValidationError, match="no puede estar vacío"):
        ChatMessage(role="user", content="   ")
