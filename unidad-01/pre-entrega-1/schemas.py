"""
Esquemas Pydantic del cliente: qué entra (mensajes y configuración) y qué sale
(respuesta, fragmentos de streaming y errores controlados).

Definirlos primero evita trabajar con diccionarios anidados sin validar.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Provider = Literal["openai", "anthropic"]
Role = Literal["system", "user", "assistant"]


# --- Entrada ------------------------------------------------------------------
class ChatMessage(BaseModel):
    """Un mensaje de la conversación."""

    model_config = ConfigDict(extra="forbid")

    role: Role
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def no_vacio(cls, valor: str) -> str:
        if not valor.strip():
            raise ValueError("El contenido del mensaje no puede estar vacío ni tener solo espacios.")
        return valor


class ModelConfig(BaseModel):
    """Configuración validada del modelo y del cliente."""

    model_config = ConfigDict(extra="forbid")

    provider: Provider
    model: str = Field(min_length=1)
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Entre 0 y 2. None = no se envía y el proveedor usa su valor por defecto.",
    )
    max_tokens: int = Field(default=1024, ge=16, le=32_000, description="Tokens máximos de la respuesta.")
    timeout: float = Field(default=60.0, gt=0, le=600, description="Segundos máximos por llamada.")
    max_retries: int = Field(default=2, ge=0, le=5, description="Reintentos ante rate limit o errores de red.")

    # Mensajes de error personalizados: se validan ANTES que las restricciones ge/le
    @field_validator("temperature", mode="before")
    @classmethod
    def validar_temperatura(cls, valor):
        if valor is None or valor == "":
            return None
        try:
            numero = float(valor)
        except (TypeError, ValueError):
            raise ValueError(f"La temperatura debe ser un número entre 0 y 2 (recibido: {valor!r}).") from None
        if not 0.0 <= numero <= 2.0:
            raise ValueError(f"La temperatura debe estar entre 0 y 2 (recibido: {numero}).")
        return numero

    @field_validator("max_tokens", mode="before")
    @classmethod
    def validar_max_tokens(cls, valor):
        try:
            numero = int(valor)
        except (TypeError, ValueError):
            raise ValueError(f"max_tokens debe ser un número entero (recibido: {valor!r}).") from None
        if numero < 16:
            raise ValueError(f"max_tokens debe ser al menos 16 para obtener una respuesta útil (recibido: {numero}).")
        if numero > 32_000:
            raise ValueError(f"max_tokens no puede superar 32000 (recibido: {numero}).")
        return numero


# --- Salida -------------------------------------------------------------------
ErrorType = Literal["rate_limit", "connection", "timeout", "authentication", "bad_request", "api_error", "unknown"]


class LLMError(BaseModel):
    """Error controlado: se devuelve en lugar de dejar que la excepción rompa el programa."""

    type: ErrorType
    message: str
    provider: Provider
    retryable: bool
    status_code: int | None = None
    attempts: int = 1


class Usage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None


class ModelResponse(BaseModel):
    """Respuesta normal (no streaming). Si algo falló, `error` viene completo y `content` vacío."""

    provider: Provider
    model: str
    content: str = ""
    finish_reason: str | None = None
    usage: Usage | None = None
    error: LLMError | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class StreamChunk(BaseModel):
    """Un fragmento del streaming. El último trae done=True (y `error` si algo falló)."""

    delta: str = ""
    done: bool = False
    error: LLMError | None = None
