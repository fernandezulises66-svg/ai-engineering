# Pre-entrega 1 · Cliente de LLM robusto y asíncrono

**Unified Async LLM Client**: una interfaz única para hablar con OpenAI o con Anthropic, con llamadas no bloqueantes (`async/await`), streaming de tokens con `yield`, validación con Pydantic y manejo controlado de errores de red y de rate limiting.

```python
async with AsyncLLMManager.from_env() as llm:          # proveedor según LLM_PROVIDER
    r = await llm.generate("¿Qué es la entropía?")     # modo normal → ModelResponse
    async for chunk in llm.stream("¿Qué es la entropía?"):   # streaming → StreamChunk
        print(chunk.delta, end="", flush=True)
```

## Arquitectura

```
main.py ──► AsyncLLMManager  (manager.py)
              │  · Factory: LLM_PROVIDER → clase del proveedor
              │  · Reintentos con backoff exponencial (respeta Retry-After)
              │  · Excepciones del SDK → LLMError controlado (nunca crashea)
              ▼
            BaseLLMClient  (clients.py, clase abstracta: generate() + stream())
              ├── OpenAIClient     → AsyncOpenAI.chat.completions.create(...)
              └── AnthropicClient  → AsyncAnthropic.messages.create / .stream(...)

schemas.py: ChatMessage · ModelConfig · ModelResponse · StreamChunk · LLMError
```

| Archivo | Contenido |
|---|---|
| `schemas.py` | Modelos Pydantic de entrada (`ChatMessage`, `ModelConfig`) y salida (`ModelResponse`, `StreamChunk`, `LLMError`). |
| `clients.py` | `BaseLLMClient` (abstracta) y las implementaciones `OpenAIClient` y `AnthropicClient` con los clientes **asíncronos** de cada SDK. |
| `manager.py` | `AsyncLLMManager`: factory por variable de entorno, `generate()`, `stream()`, clasificación de errores y reintentos. |
| `main.py` | Script de validación: la misma pregunta en modo normal y en streaming, más una prueba de error controlado. |
| `tests/test_offline.py` | 22 tests con los SDKs reales y una red simulada (`httpx.MockTransport`): respuestas, streaming SSE, 429, 401, caída de red. No necesitan API keys. |

## Cómo cumple cada requisito

| Requisito | Implementación |
|---|---|
| **Intercambiabilidad** | `BaseLLMClient` define la interfaz común. `AsyncLLMManager.from_env()` elige `OpenAIClient` o `AnthropicClient` según `LLM_PROVIDER` (patrón Factory + Strategy). El resto del código no cambia al cambiar de proveedor. |
| **Asincronía** | Se usan `AsyncOpenAI` y `AsyncAnthropic` con `await`. Ninguna llamada usa el cliente síncrono, así que el event loop nunca se bloquea. |
| **Streaming** | `stream()` es un generador asíncrono: recorre el stream del SDK con `async for` y hace `yield` de cada fragmento apenas llega. `main.py` mide el tiempo hasta el primer token (TTFT). |
| **Validación (Pydantic)** | `ChatMessage` valida rol y contenido no vacío. `ModelConfig` valida `temperature` (0 a 2, `ge`/`le`) y `max_tokens` (16 a 32000) con **mensajes de error personalizados**, además de `timeout` y `max_retries`. |
| **Errores de red y rate limit** | `RateLimitError`, `APIConnectionError`, `APITimeoutError`, `AuthenticationError`, `BadRequestError` y `APIStatusError` de ambos SDKs se convierten en un `LLMError` con `type`, `status_code`, `retryable` y `attempts`. Se reintentan solo los errores transitorios (429, red, timeout, 5xx), con backoff exponencial + jitter y respetando el header `Retry-After`. |

### Decisiones de diseño

- **Errores como datos, no como excepciones:** `generate()` siempre devuelve un `ModelResponse` (con `ok=False` y `error` si falló), y `stream()` termina con un `StreamChunk(done=True, error=...)`. Quien usa el cliente revisa un campo, en lugar de envolver cada llamada en `try/except`.
- **Una sola política de reintentos:** los SDKs se crean con `max_retries=0` y los reintentos los maneja el manager, así la lógica es visible, se registra en los logs y es igual para los dos proveedores.
- **Streaming seguro:** en `stream()` solo se reintenta si el error ocurre *antes* del primer fragmento. Si ya se mostró texto, reintentar lo duplicaría.
- **Anthropic recibe el `system` aparte:** los mensajes con rol `system` se extraen y se envían en el parámetro `system`, como pide su API.
- **`temperature` opcional:** si no se configura, no se envía. Los modelos de razonamiento recientes (Claude 5.x y la familia GPT-6) rechazan valores distintos al de fábrica; en ese caso el 400 se devuelve como error controlado de tipo `bad_request`.

## Cómo ejecutarlo

Requiere **Python 3.12+**.

```bash
cd unidad-01/pre-entrega-1
python -m venv .venv
.venv\Scripts\activate          # Windows  (en Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # y completá las variables (ver abajo)
```

```bash
python main.py                      # proveedor de LLM_PROVIDER: modo normal + streaming + error controlado
python main.py --provider openai    # fuerza un proveedor
python main.py --both               # prueba OpenAI y Anthropic, uno después del otro
pytest -v                           # tests offline (sin API keys ni red)
```

## Variables de entorno

| Variable | Obligatoria | Descripción |
|---|---|---|
| `LLM_PROVIDER` | No (por defecto `anthropic`) | Proveedor activo: `openai` o `anthropic`. |
| `OPENAI_API_KEY` | Si usás OpenAI | Key de [platform.openai.com/api-keys](https://platform.openai.com/api-keys). |
| `ANTHROPIC_API_KEY` | Si usás Anthropic | Key de [platform.claude.com/settings/keys](https://platform.claude.com/settings/keys). |
| `ANTHROPIC_WORKSPACE_ID` | No | Solo si la key de Anthropic no está asociada a un workspace. |
| `OPENAI_MODEL` / `ANTHROPIC_MODEL` | No | Modelo a usar (por defecto `gpt-6-luna` y `claude-haiku-5-5`). |
| `LLM_TEMPERATURE` | No | Entre 0 y 2. Vacío = valor por defecto del proveedor. |
| `LLM_MAX_TOKENS` | No (por defecto 1024) | Tokens máximos de la respuesta (16 a 32000). |
| `LLM_MAX_RETRIES` | No (por defecto 2) | Reintentos ante rate limit o errores de red. |

## Ejemplo: error controlado

Con una API key inválida, el programa no se cae (salida real contra la API de Anthropic):

```
--- 1. Modo normal (generate) ---
22:29:22 | ERROR   | [anthropic] error controlado: authentication
✗ Error controlado: authentication — Error code: 401 - {... 'message': 'invalid x-api-key'}

--- 2. Streaming (stream con yield) ---
22:29:22 | ERROR   | [anthropic] stream interrumpido: authentication
✗ Error controlado: authentication — Error code: 401 - {...}

--- 3. Error controlado (API key inválida) ---
ok=False · tipo=authentication · status=401
→ El programa sigue funcionando: el error se devolvió como dato, no como excepción.
```
