"""
AsyncLLMManager: punto de entrada único del "Unified Async LLM Client".

- Factory: elige OpenAIClient o AnthropicClient según la variable LLM_PROVIDER.
- generate(): respuesta completa, con reintentos y backoff exponencial.
- stream(): generador asíncrono que hace `yield` de cada fragmento apenas llega.
- Nunca deja escapar una excepción de la API: la convierte en un LLMError controlado.
"""

import asyncio
import logging
import os
import random
from collections.abc import AsyncIterator, Awaitable, Callable

import anthropic
import openai
from dotenv import find_dotenv, load_dotenv

from clients import AnthropicClient, BaseLLMClient, OpenAIClient
from schemas import ChatMessage, LLMError, ModelConfig, ModelResponse, StreamChunk

logger = logging.getLogger("llm")

# Registro de proveedores (Factory): agregar uno nuevo = una línea acá
PROVIDERS: dict[str, type[BaseLLMClient]] = {"openai": OpenAIClient, "anthropic": AnthropicClient}
DEFAULT_MODELS = {"openai": "gpt-6-luna", "anthropic": "claude-haiku-5-5"}
MODEL_ENV_VARS = {"openai": "OPENAI_MODEL", "anthropic": "ANTHROPIC_MODEL"}


# --- Clasificación de errores -------------------------------------------------
def classify_error(exc: Exception, provider: str) -> LLMError:
    """Traduce las excepciones de ambos SDKs a un LLMError con tipo y si conviene reintentar."""

    def err(type_, retryable, status=None):
        return LLMError(type=type_, message=str(exc)[:500], provider=provider, retryable=retryable, status_code=status)

    status = getattr(exc, "status_code", None)
    # El orden importa: APITimeoutError es una subclase de APIConnectionError
    if isinstance(exc, (openai.RateLimitError, anthropic.RateLimitError)):
        return err("rate_limit", True, status)
    if isinstance(exc, (openai.APITimeoutError, anthropic.APITimeoutError, asyncio.TimeoutError)):
        return err("timeout", True)
    if isinstance(exc, (openai.APIConnectionError, anthropic.APIConnectionError)):
        return err("connection", True)
    if isinstance(exc, (openai.AuthenticationError, anthropic.AuthenticationError,
                        openai.PermissionDeniedError, anthropic.PermissionDeniedError)):
        return err("authentication", False, status)
    if isinstance(exc, (openai.BadRequestError, anthropic.BadRequestError, openai.NotFoundError,
                        anthropic.NotFoundError, openai.UnprocessableEntityError, anthropic.UnprocessableEntityError)):
        return err("bad_request", False, status)
    if isinstance(exc, (openai.APIStatusError, anthropic.APIStatusError)):
        return err("api_error", (status or 0) >= 500, status)  # 500, 503, 529 (overloaded): reintentables
    return err("unknown", False)


def _retry_after(exc: Exception) -> float | None:
    """Si la API indica cuánto esperar (header Retry-After), se respeta."""
    response = getattr(exc, "response", None)
    try:
        return float(response.headers.get("retry-after")) if response is not None else None
    except (TypeError, ValueError):
        return None


# --- Manager ------------------------------------------------------------------
class AsyncLLMManager:
    def __init__(self, config: ModelConfig, client: BaseLLMClient | None = None,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
        if config.provider not in PROVIDERS:
            raise ValueError(f"Proveedor no soportado: {config.provider}. Opciones: {', '.join(PROVIDERS)}")
        self.config = config
        self.client = client or PROVIDERS[config.provider](config)
        self._sleep = sleep  # inyectable para que los tests no esperen de verdad

    @classmethod
    def from_env(cls, provider: str | None = None, **overrides) -> "AsyncLLMManager":
        """Arma la configuración desde variables de entorno (.env) y la valida con Pydantic."""
        load_dotenv(find_dotenv())
        provider = (provider or os.getenv("LLM_PROVIDER", "anthropic")).strip().lower()
        if provider not in PROVIDERS:
            raise ValueError(f"LLM_PROVIDER inválido: {provider!r}. Opciones: {', '.join(PROVIDERS)}")
        values = {
            "provider": provider,
            "model": os.getenv(MODEL_ENV_VARS[provider]) or DEFAULT_MODELS[provider],
            "temperature": os.getenv("LLM_TEMPERATURE") or None,
            "max_tokens": os.getenv("LLM_MAX_TOKENS") or 1024,
            "max_retries": os.getenv("LLM_MAX_RETRIES") or 2,
            **overrides,
        }
        return cls(ModelConfig(**values))

    @staticmethod
    def _normalize(messages: list[ChatMessage] | str) -> list[ChatMessage]:
        if isinstance(messages, str):
            return [ChatMessage(role="user", content=messages)]
        if not messages:
            raise ValueError("Se necesita al menos un mensaje.")
        return messages

    def _backoff(self, attempt: int, exc: Exception) -> float:
        return _retry_after(exc) or min(2 ** attempt, 20) + random.uniform(0, 0.5)

    async def generate(self, messages: list[ChatMessage] | str) -> ModelResponse:
        msgs = self._normalize(messages)
        for attempt in range(self.config.max_retries + 1):
            try:
                response = await self.client.generate(msgs)
                logger.info("[%s] respuesta OK (%d caracteres)", self.config.provider, len(response.content))
                return response
            except Exception as exc:  # cualquier error se convierte en un LLMError controlado
                error = classify_error(exc, self.config.provider)
                error.attempts = attempt + 1
                if error.retryable and attempt < self.config.max_retries:
                    wait = self._backoff(attempt, exc)
                    logger.warning("[%s] %s (intento %d) → reintento en %.1fs",
                                   self.config.provider, error.type, attempt + 1, wait)
                    await self._sleep(wait)
                    continue
                logger.error("[%s] error controlado: %s", self.config.provider, error.type)
                return ModelResponse(provider=self.config.provider, model=self.config.model, error=error)

    async def stream(self, messages: list[ChatMessage] | str) -> AsyncIterator[StreamChunk]:
        """Generador asíncrono: `async for chunk in manager.stream(...)`.

        Reintenta solo si el error ocurre ANTES del primer fragmento (si ya se mostró
        texto, reintentar lo duplicaría). El último chunk trae done=True.
        """
        msgs = self._normalize(messages)
        for attempt in range(self.config.max_retries + 1):
            emitted = False
            try:
                async for text in self.client.stream(msgs):
                    emitted = True
                    yield StreamChunk(delta=text)
                yield StreamChunk(done=True)
                return
            except Exception as exc:
                error = classify_error(exc, self.config.provider)
                error.attempts = attempt + 1
                if error.retryable and not emitted and attempt < self.config.max_retries:
                    wait = self._backoff(attempt, exc)
                    logger.warning("[%s] %s al iniciar el stream → reintento en %.1fs",
                                   self.config.provider, error.type, wait)
                    await self._sleep(wait)
                    continue
                logger.error("[%s] stream interrumpido: %s", self.config.provider, error.type)
                yield StreamChunk(done=True, error=error)
                return

    async def aclose(self) -> None:
        await self.client.aclose()

    async def __aenter__(self) -> "AsyncLLMManager":
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.aclose()
