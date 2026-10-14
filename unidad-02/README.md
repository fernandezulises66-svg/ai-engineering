# Unidad 2 · Arquitectura de cadenas con LCEL y Runnables

| Entrega | Qué es | Carpeta |
|---|---|---|
| Ejercicio | Refactorización de una llamada al SDK a una cadena LCEL asíncrona | [ejercicio-lcel](./ejercicio-lcel) |
| Pre-entrega 2 | Pipeline de extracción de entidades técnicas validado con Pydantic y con reintentos | [pre-entrega-2](./pre-entrega-2) |

Proveedor: **Anthropic** (`ChatAnthropic`, modelo por defecto `claude-haiku-5-5`).

## Setup (una sola vez)

```bash
cd unidad-02
python -m venv .venv
.venv\Scripts\activate          # Windows  (en Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # y completá ANTHROPIC_API_KEY
```
