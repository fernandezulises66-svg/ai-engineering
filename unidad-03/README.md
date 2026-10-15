# Unidad 3 · Embeddings y similitud

| Entrega | Qué es | Carpeta |
|---|---|---|
| Ejercicio (PDF) | Relevancia semántica vs. coincidencia de palabras clave, código de similitud coseno y diagrama de búsqueda semántica | [ejercicio-embeddings](./ejercicio-embeddings) |
| Pre-entrega 3 | Sistema RAG local: ingesta en ChromaDB + cadena LCEL asíncrona con respuestas grounded | [pre-entrega-3](./pre-entrega-3) |

Proveedores: **OpenAI** para embeddings (`text-embedding-3-small`) y **Anthropic** para generación (`claude-haiku-5-5`).

## Setup (una sola vez)

```bash
cd unidad-03
python -m venv .venv
.venv\Scripts\activate          # Windows  (en Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # y completá ANTHROPIC_API_KEY y OPENAI_API_KEY
```
