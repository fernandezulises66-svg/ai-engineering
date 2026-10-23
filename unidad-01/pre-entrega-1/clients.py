"""
Clientes por proveedor (patrón Strategy): misma interfaz, distinta implementación.

    BaseLLMClient (abstracta)
      ├── OpenAIClient     → AsyncOpenAI    (chat.completions)
      └── AnthropicClient  → AsyncAnthropic (messages)

Ambos usan los clientes ASÍNCRONOS de cada SDK, así ninguna llamada bloquea
el event loop. Los reintentos los maneja AsyncLLMManager (por eso max_retries=0
en el SDK): así la política de reintentos es una sola y queda visible en el código.
"""

import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from schemas import ChatMessage, ModelConfig, ModelResponse, Usage


class BaseLLMClient(ABC):
    """Interfaz común para cualquier proveedor."""

    provider: str

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abstractmethod
    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        """Devuelve la respuesta completa."""

    @abstractmethod
    def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        """Generador asíncrono: hace `yield` de cada fragmento de texto apenas llega."""

    @abstractmethod
    async def aclose(self) -> None:
        """Cierra las conexiones HTTP del cliente."""


class OpenAIClient(BaseLLMClient):
    provider = "openai"

    def __init__(self, config: ModelConfig, client: AsyncOpenAI | None = None) -> None:
        super().__init__(config)
        # Lee OPENAI_API_KEY del entorno
        self._client = client or AsyncOpenAI(timeout=config.timeout, max_retries=0)

    def _params(self, messages: list[ChatMessage]) -> dict:
        params = {
            "model": self.config.model,
            "messages": [m.model_dump() for m in messages],
            "max_completion_tokens": self.config.max_tokens,
        }
        if self.config.temperature is not None:
            params["temperature"] = self.config.temperature
        return params

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        r = await self._client.chat.completions.create(**self._params(messages))
        choice = r.choices[0]
        return ModelResponse(
            provider="openai",
            model=r.model,
            content=choice.message.content or "",
            finish_reason=choice.finish_reason,
            usage=Usage(input_tokens=r.usage.prompt_tokens, output_tokens=r.usage.completion_tokens) if r.usage else None,
        )

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(**self._params(messages), stream=True)
        async for chunk in stream:  # cada chunk llega por la red apenas el modelo lo genera
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def aclose(self) -> None:
        await self._client.close()


class AnthropicClient(BaseLLMClient):
    provider = "anthropic"

    def __init__(self, config: ModelConfig, client: AsyncAnthropic | None = None) -> None:
        super().__init__(config)
        headers = {}
        # Keys no asociadas a un workspace necesitan indicar cuál usar
        if workspace := os.getenv("ANTHROPIC_WORKSPACE_ID"):
            headers["anthropic-workspace-id"] = workspace
        # Lee ANTHROPIC_API_KEY del entorno
        self._client = client or AsyncAnthropic(timeout=config.timeout, max_retries=0, default_headers=headers)

    def _params(self, messages: list[ChatMessage]) -> dict:
        # Anthropic recibe las instrucciones de sistema aparte, no como un mensaje más
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        params = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": [m.model_dump() for m in messages if m.role != "system"],
        }
        if system:
            params["system"] = system
        if self.config.temperature is not None:
            params["temperature"] = self.config.temperature
        return params

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        r = await self._client.messages.create(**self._params(messages))
        return ModelResponse(
            provider="anthropic",
            model=r.model,
            content="".join(block.text for block in r.content if block.type == "text"),
            finish_reason=r.stop_reason,
            usage=Usage(input_tokens=r.usage.input_tokens, output_tokens=r.usage.output_tokens),
        )

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        async with self._client.messages.stream(**self._params(messages)) as stream:
            async for text in stream.text_stream:  # solo los fragmentos de texto
                yield text

    async def aclose(self) -> None:
        await self._client.close()
