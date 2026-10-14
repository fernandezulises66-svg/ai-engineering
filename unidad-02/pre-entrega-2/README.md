# Pre-entrega 2 · Pipeline de extracción de entidades técnicas

Recibe un texto técnico sin procesar (un log de error, una descripción de arquitectura, un incidente) y devuelve un objeto **validado con Pydantic**. Ejemplo de salida para el log de error de `test_pipeline.py`:

```json
{
  "tecnologias": ["FastAPI", "Kubernetes", "PostgreSQL", "psycopg2", "Redis"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "Los pods de FastAPI devuelven 503 en /checkout porque PostgreSQL rechaza conexiones nuevas (límite alcanzado), con Redis cerca de su límite de memoria."
}
```

## Arquitectura

```
{"texto": ...}
   │
   ▼
ChatPromptTemplate (system + human, instrucciones de formato como variable parcial)
   │
   ▼
ChatAnthropic.with_structured_output(ExtraccionTecnica, include_raw=True)
   │
   ▼
validar_salida ── stop_reason == "max_tokens"  → SalidaIncompletaError ─┐
   │           ── JSON mal formado / campo inválido → ValidationError ──┤ .with_retry()
   │           ── sin estructura               → OutputParserException ┘ (hasta 3 intentos)
   ▼
ExtraccionTecnica (objeto Pydantic validado)
```

| Archivo | Contenido |
|---|---|
| `schemas.py` | Modelo Pydantic `ExtraccionTecnica`: `tecnologias` (lista no vacía, sin duplicados), `nivel_de_criticidad` (enum `baja`/`media`/`alta`) y `resumen_tecnico` (20–400 caracteres). |
| `chain.py` | Modelo, prompt, cadena LCEL `prompt \| model.with_structured_output(Schema)`, validación, `.with_retry()` y la función asíncrona `process_text()`. |
| `test_pipeline.py` | Mini-script asíncrono que procesa en paralelo tres textos contra la API real (uno de ellos ambiguo, como prueba de estrés). |
| `tests/test_offline.py` | Tests con `pytest` que **no necesitan API key**: un modelo falso simula JSON inválido, una respuesta cortada y fallos repetidos para verificar los reintentos. |

## Decisiones de diseño

- **`with_structured_output` + `include_raw=True`**: además del objeto parseado, devuelve el mensaje crudo. Así se puede revisar el `stop_reason` y detectar una respuesta **cortada por falta de tokens** antes de intentar convertirla.
- **Dos niveles de reintento**: `max_retries=2` en `ChatAnthropic` cubre los errores de red y los 429. `.with_retry()` cubre los errores de **contenido** (JSON mal formado, enum inválido, lista vacía, salida incompleta) y vuelve a llamar al modelo, con backoff exponencial y jitter.
- **El prompt sin f-strings**: las instrucciones de formato se inyectan con `.partial()`, así quedan separadas del texto del usuario y se pueden cambiar sin tocar la cadena.
- **Modelo `claude-haiku-5-5`**: rápido y barato, pensado para tareas de extracción. Se cambia con `ANTHROPIC_MODEL` en el `.env`.
- **Sin `temperature`**: los modelos Claude 5.x devuelven error 400 si se envía. El código solo la manda si se configura `ANTHROPIC_TEMPERATURE`, para usar un modelo anterior que la acepte.

## Cómo correrlo

Desde la carpeta `unidad-02`:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (en Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # y completá ANTHROPIC_API_KEY
cd pre-entrega-2
```

```bash
python test_pipeline.py   # prueba real contra la API (3 textos en paralelo)
python chain.py           # un solo ejemplo
pytest -v                 # tests offline de validación y reintentos (sin API key)
```

## Logs de ejemplo

Un reintento por JSON inválido se ve así:

```
21:55:30 | INFO    | pipeline | Procesando texto (40 caracteres)
21:55:30 | INFO    | pipeline | Llamando al modelo...
21:55:30 | WARNING | pipeline | Salida inválida (ValidationError): 1 validation error for ExtraccionTecnica
nivel_de_criticidad
  Input should be 'baja', 'media' or 'alta' [type=enum, input_value='urgente', input_type=str] → reintento
21:55:31 | INFO    | pipeline | Llamando al modelo...
21:55:31 | INFO    | pipeline | Validación OK (stop_reason=tool_use)
21:55:31 | INFO    | pipeline | Extracción completada: 2 tecnologías, criticidad alta
```
