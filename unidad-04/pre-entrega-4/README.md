# Pre-entrega 4 · Sistema RAG escalable en la nube con Pinecone

Módulo de recuperación sobre la documentación técnica del ERP de **OLA Muebles**: los documentos se indexan en **Pinecone Serverless** y se consultan con un **recuperador híbrido**, que combina búsqueda vectorial (significado) con **BM25** (palabras exactas). Un script de evaluación mide **Precision@5**, **Recall@5** y MRR sobre un *golden set* de preguntas.

> Los documentos de `data/` son **ficticios**, creados para este ejercicio.

## Arquitectura

```
INGESTA (setup_index.py + ingest.py)
  data/*.md  ┐
  data/*.json├─► loaders (fuente, página, categoría, tags) ─► RecursiveCharacterTextSplitter (600 tokens, overlap 80)
  data/*.pdf ┘        ─► OpenAIEmbeddings (text-embedding-3-small, 1536 dim.)
                      ─► Pinecone Serverless · índice INDEX_NAME · namespace "erp-docs-v1"
                         metadata = {doc_id, title, source, page, category, tags, chunk_id, text}

CONSULTA (rag_system.py · clase RAGSystem)
  query ─┬─► BM25Retriever (chunks locales, tokenizador para español técnico) ─ top 10 ─┐
         └─► PineconeVectorStore (similitud coseno en el namespace)        ─ top 10 ─┴─► EnsembleRetriever (RRF)
                                                                                          │
         ¿la consulta trae un identificador (ERR-AFIP-102, OLA-RIB-0042)? ── sí ─► esos chunks primero ─► top 5

EVALUACIÓN (evaluate.py)
  golden_set.json ─► BM25 vs. vectorial vs. híbrido vs. híbrido + exactos ─► Precision@5 · Recall@5 · MRR
```

| Archivo | Contenido |
|---|---|
| `data/` | Dataset: 6 documentos Markdown, 1 JSON (catálogo de códigos de error) y 1 PDF de 3 páginas (integración con AFIP). `_metadata.json` tiene los metadatos del PDF. |
| `config.py` | Nombre del índice, namespace, región, modelo de embeddings y su dimensión, parámetros de chunking y de recuperación. |
| `setup_index.py` | Verifica si el índice existe y lo crea en modo Serverless. Si ya existe, valida que su dimensión coincida con la del modelo. |
| `ingest.py` | Carga los 3 formatos, divide en chunks y los sube a Pinecone con metadatos y el texto original. |
| `rag_system.py` | Clase `RAGSystem`: `EnsembleRetriever` que fusiona BM25 y Pinecone (RRF), con prioridad a identificadores exactos, y devuelve el top-5. |
| `evaluate.py` + `golden_set.json` | Benchmark de 5 preguntas con su documento fuente conocido y el reporte de métricas. |
| `tests/test_offline.py` | 15 tests sin API keys (Pinecone falso, embeddings determinísticos). |

## Decisiones de diseño

- **El texto original vive en la metadata de Pinecone** (`text`): cada resultado trae el contenido, la fuente y la página, sin consultar otra base de datos.
- **Metadatos avanzados:** `doc_id`, `title`, `source`, `page` (número de página real en el PDF), `category` y `tags`. Permiten filtrar (por ejemplo, solo `category = "fiscal"`) y citar la fuente exacta.
- **Namespace propio** (`erp-docs-v1`): aísla este dataset de otros en el mismo índice. Para una versión nueva del dataset se usa otro namespace, sin mezclar resultados.
- **Ids estables por chunk** (`integracion-afip#p3-c1`): volver a ingerir sobrescribe en lugar de duplicar. `--reset` vacía el namespace antes de subir.
- **Validación de dimensiones:** `setup_index.py` frena con un mensaje claro si el índice existente no tiene la dimensión del modelo de embeddings (el error típico de subir vectores de 1536 a un índice de 768).
- **Chunks de 600 tokens con 80 de solapamiento,** cortando primero en los títulos del Markdown. El JSON se agrupa por módulo, para que cada chunk tenga contexto suficiente.
- **BM25 adaptado al español técnico:** el tokenizador quita acentos y palabras vacías, y conserva los códigos con guiones como un solo término (`err-afip-102`), agregando también sus partes (`err`, `afip`, `102`). Así matchea tanto "ERR-AFIP-102" como "error 102 de AFIP".
- **Fusión por Reciprocal Rank Fusion (RRF)** con pesos 50/50: cada recuperador aporta sus 10 mejores candidatos y se combinan por posición, no por puntaje (los puntajes de BM25 y del coseno no son comparables entre sí). La deduplicación se hace por `chunk_id`.
- **El PDF repara códigos cortados:** si la extracción de texto parte un código en dos líneas ("ERR-" / "AFIP-102"), el loader lo vuelve a unir.

## Cómo replicar el índice

### 1. Crear las cuentas y las keys

- **Pinecone:** creá una cuenta en [app.pinecone.io](https://app.pinecone.io) (el plan gratuito *Starter* alcanza) y copiá la API key desde **API Keys**.
- **OpenAI:** creá una key en [platform.openai.com/api-keys](https://platform.openai.com/api-keys) (se usa solo para los embeddings).

### 2. Instalar y configurar

Requiere Python 3.12+.

```bash
cd unidad-04/pre-entrega-4
python -m venv .venv
.venv\Scripts\activate          # Windows  (en Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # completá PINECONE_API_KEY, OPENAI_API_KEY e INDEX_NAME
```

### 3. Crear el índice, ingerir y evaluar

```bash
python setup_index.py     # crea el índice serverless (1536 dim., coseno, aws/us-east-1) si no existe
python ingest.py          # sube los chunks al namespace erp-docs-v1
python rag_system.py "¿Qué significa el error ERR-AFIP-102?"   # una consulta de prueba
python evaluate.py        # métricas de BM25, vectorial, híbrido e híbrido + exactos
pytest -v                 # tests offline
```

`python evaluate.py --modes bm25` corre la evaluación solo con BM25, sin Pinecone ni OpenAI.

## Evaluación

El golden set tiene 5 preguntas de distinto tipo, con el documento fuente conocido de antemano:

| # | Pregunta | Documento esperado | Tipo |
|---|---|---|---|
| q1 | ¿Qué significa el error ERR-AFIP-102 y cómo se soluciona? | `integracion-afip` | código técnico exacto |
| q2 | ¿Cuánto tiempo puede trabajar un usuario antes de tener que volver a ingresar su contraseña? | `autenticacion` | paráfrasis |
| q3 | ¿En qué momento un pedido de la tienda online se transforma en una venta del sistema? | `tiendanube` | conceptual |
| q4 | Subí un cambio al backend pero no se ve en producción, ¿qué me faltó hacer? | `arquitectura` | lenguaje coloquial |
| q5 | ¿Qué quiere decir el código OLA-RIB-0042? | `inventario` | identificador propio |

**Definiciones** (a nivel de documento fuente, k = 5):

- **Recall@5:** 1 si algún chunk del documento esperado está entre los 5 recuperados, 0 si no.
- **Precision@5:** proporción de los 5 chunks que vienen de un documento relevante para la pregunta.
- **MRR:** 1 / posición del primer chunk del documento esperado.

**Techo de Precision@5:** cada documento tiene pocos chunks (el dataset completo tiene 20), así que aunque el recuperador sea perfecto, varias preguntas no pueden llenar los 5 lugares con chunks relevantes. El techo promedio es **0,64**, y `evaluate.py` lo informa para leer la precisión en contexto.

### Resultados (26/10/2026, Pinecone + OpenAI)

| Recuperador | Precision@5 | Recall@5 | MRR |
|---|---|---|---|
| BM25 (léxico) | 0,40 | 1,00 | 0,90 |
| Vectorial (Pinecone) | 0,40 | 0,80 | 0,70 |
| Híbrido (RRF) | **0,48** | **1,00** | 0,74 |
| **Híbrido + exactos** (modo por defecto de `RAGSystem`) | **0,48** | **1,00** | **0,90** |

El reporte completo de consola, con el detalle por pregunta, está en [`resultados_evaluacion.md`](./resultados_evaluacion.md).

| Pregunta | BM25 | Vectorial | Híbrido (RRF) | Híbrido + exactos |
|---|---|---|---|---|
| q1 · ERR-AFIP-102 | #2 | #1 | #2 | #2 |
| q2 · duración de la sesión (paráfrasis) | #1 | #2 | #1 | #1 |
| q3 · pedido → venta | #1 | #1 | #1 | #1 |
| q4 · backend sin reiniciar (coloquial) | #1 | #1 | #1 | #1 |
| q5 · OLA-RIB-0042 | #1 | **no aparece** | #5 | **#1** |

*Posición del primer chunk del documento esperado. En q1, el puesto 1 lo ocupa el catálogo de códigos de error, que también es relevante (por eso la precisión es 0,8).*

### Análisis

- **Cada recuperador tiene un punto ciego.** La búsqueda vectorial no encuentra el SKU `OLA-RIB-0042` (q5): un código arbitrario no tiene un "significado" que el modelo de embeddings haya aprendido. BM25 lo encuentra al instante, pero en la paráfrasis de q2 trae menos chunks útiles (precisión 0,4 contra 0,6 de la vectorial).
- **El híbrido mejora la precisión** (0,48 contra 0,40 de cada uno por separado) porque combina lo mejor de ambos: en q2 y q3 suma chunks relevantes que uno solo no traía.
- **Pero RRF premia el consenso.** Un chunk que BM25 pone 1.º y la búsqueda vectorial no trae suma 0,5/61 ≈ 0,008. Uno que está 3.º y 5.º en las dos listas suma 0,5/63 + 0,5/65 ≈ 0,016. En q5, eso empujó al documento correcto al puesto 5 (y en una primera corrida, afuera del top-5).
- **La corrección: prioridad a identificadores exactos.** Si la consulta contiene un código (letras y números unidos por guiones, con al menos un dígito), los chunks que lo contienen literalmente van primero y después sigue el ranking híbrido. Sin identificadores, el resultado es idéntico al híbrido puro (hay un test que lo verifica). Con esto, el MRR sube de 0,74 a 0,90 y q5 queda en el puesto 1 de forma estable.
- **Limitación:** son 5 preguntas, así que cada una mueve las métricas un 20 %. Las conclusiones sirven para entender el comportamiento, no como estadística. El siguiente paso sería ampliar el golden set a 30 o 50 preguntas.
