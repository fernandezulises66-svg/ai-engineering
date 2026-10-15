# Pre-entrega 3 · Sistema de recuperación semántica local (RAG)

Asistente que responde preguntas sobre **OLA Muebles** (envíos, garantía, materiales y uso interno del ERP) usando **solo** la información de sus documentos. Si la respuesta no está en los documentos, responde *"No lo sé"* en lugar de inventarla.

> Los documentos de `data/` son **ficticios**, creados para este ejercicio. No representan condiciones comerciales reales.

## Arquitectura

```
INGESTA (ingest.py) ─ se ejecuta una vez, o cuando cambian los documentos
  data/*.md ─► RecursiveCharacterTextSplitter (500 tokens, overlap 50, tiktoken)
            ─► OpenAIEmbeddings (text-embedding-3-small)
            ─► ChromaDB persistente en ./vectorstore (similitud coseno)

CONSULTA (rag.py) ─ get_rag_response(query)
  {"pregunta"} ─► embedding de la pregunta (mismo modelo) ─► top-4 fragmentos en Chroma
              ─► contexto numerado [1]..[4] ─► prompt "filtro de veracidad"
              ─► ChatAnthropic (claude-haiku-5-5) ─► PydanticOutputParser
              ─► RAGResponse {respuesta, encontrada_en_contexto, referencias}
```

| Archivo | Contenido |
|---|---|
| `data/` | Dataset de ejemplo: 4 documentos en Markdown (envíos, garantía y devoluciones, catálogo y cuidados, manual del ERP). |
| `config.py` | Configuración central: rutas, parámetros de chunking, `top_k` y **un único** modelo de embeddings para indexar y consultar. |
| `ingest.py` | Carga, chunking por tokens y persistencia en ChromaDB. Evita reindexar si la base ya está al día. |
| `rag.py` | Cadena RAG en LCEL y la función asíncrona `get_rag_response(query)`. |
| `demo.py` | Las pruebas pedidas: preguntas con respuesta en los documentos y preguntas trampa. |
| `tests/test_offline.py` | 8 tests con `pytest` que no necesitan API keys (embeddings y LLM falsos). |

## Decisiones de diseño

- **Chunking por tokens**: el tamaño de cada chunk se mide en tokens (500, overlap 50) con `tiktoken`, no en caracteres. Los separadores priorizan los títulos `##` del Markdown, para no cortar una sección por la mitad.
- **Persistencia inteligente**: la colección guarda una huella (hash) de los documentos, de los parámetros de chunking y del modelo de embeddings. `ingest()` solo reindexa si alguno cambió (o con `--force`), y así no se vuelve a pagar la indexación en cada ejecución.
- **Protección contra embeddings no coincidentes**: si la base se indexó con otro modelo de embeddings, se detecta por la metadata y se reindexa, en lugar de comparar vectores de espacios distintos.
- **`top_k = 4`**: suficiente contexto sin caer en el "lost in the middle".
- **Referencias verificables**: el modelo solo devuelve los *números* de los fragmentos que usó. Las referencias (archivo, chunk, score, extracto) se arman desde los documentos recuperados, así que no puede inventar una fuente.
- **Doble control anti-alucinación**: el prompt obliga a responder *"No lo sé"* si falta información, y el código descarta las referencias cuando `encontrada_en_contexto` es `false`.
- **Resiliencia**: si el modelo devuelve un JSON inválido, `PydanticOutputParser` falla y `.with_retry()` vuelve a intentar (hasta 3 veces).

## Cómo ejecutarlo

Con el entorno de `unidad-03` configurado (ver [README de la unidad](../README.md)):

```bash
cd pre-entrega-3
python ingest.py                 # indexa los documentos (la 2.ª vez detecta que ya está al día)
python demo.py                   # preguntas con respuesta + preguntas trampa
python rag.py "¿Cuánto cuesta el envío a CABA?"   # una consulta suelta
pytest -v                        # tests offline (sin API keys)
```

## Pruebas: respuesta en los documentos vs. pregunta trampa

`demo.py` corre en paralelo cuatro preguntas: dos cuya respuesta está en los documentos (garantía de una bisagra y arrepentimiento en muebles a medida) y dos trampa (cuotas sin interés y cantidad de empleados), que no figuran en ningún documento. Para cada una verifica que `encontrada_en_contexto` tenga el valor esperado.

### Ejemplo de salida esperada

```json
{
  "pregunta": "Se me rompió una bisagra del rack a los 4 meses. ¿Está en garantía y cómo hago el reclamo?",
  "respuesta": "Sí: los herrajes, como las bisagras, tienen 6 meses de garantía...",
  "encontrada_en_contexto": true,
  "referencias": [
    {"fuente": "02-garantia-y-devoluciones.md", "chunk": 1, "score": 0.61, "...": "..."}
  ]
}
```

```json
{
  "pregunta": "¿En cuántas cuotas sin interés puedo pagar con tarjeta de crédito?",
  "respuesta": "No lo sé: esa información no está en la documentación disponible.",
  "encontrada_en_contexto": false,
  "referencias": []
}
```
