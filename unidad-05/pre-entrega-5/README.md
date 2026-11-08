# Pre-entrega 5 · Agente de razonamiento cíclico con memoria persistente

Agente **ReAct** construido con **LangGraph** que responde consultas sobre clientes y pedidos del ERP de OLA Muebles. El agente decide solo qué herramienta usar y cuántas veces, se corrige cuando una herramienta devuelve un error y **recuerda la conversación** por `thread_id`, incluso después de reiniciar el programa.

> Los datos de la base `erp_demo.db` son **ficticios**, creados para este ejercicio.

## Arquitectura

```
                 ┌──────────────── sin tool_calls ───────────────► END
                 │
 START ──► agent (LLM + bind_tools) ──(tools_condition)──► tools (ToolNode)
                 ▲                                              │
                 └──────────── resultado como ToolMessage ──────┘

 Estado: AgentState(MessagesState) · reducer add_messages
 Persistencia: AsyncSqliteSaver → checkpoints.sqlite (una conversación por thread_id)
 Límite: recursion_limit = 10 pasos por turno
```

| Archivo | Contenido |
|---|---|
| `agent.py` | `AgentState` (hereda de `MessagesState`), el `StateGraph` con el nodo del modelo, el `ToolNode` y la arista condicional `tools_condition`. |
| `tools.py` | Las 3 herramientas propias con `@tool` y docstrings detallados: `buscar_cliente`, `buscar_pedidos`, `detalle_pedido`. |
| `db.py` | Base SQLite de demostración (clientes, pedidos, ítems), con consultas asíncronas mediante `aiosqlite`. |
| `runner.py` | Ejecuta un turno con `astream` y arma la traza ReAct (decisión → herramienta → resultado → respuesta). |
| `demo.py` | Prueba de ejecución con 4 escenarios; genera `trazas/traza_ejemplo.json` y `.log`. |
| `main.py` | Chat interactivo por consola con memoria persistente (`--thread`). |
| `config.py` | Modelo (Anthropic u OpenAI), rutas, `recursion_limit` y tamaño del contexto. |
| `tests/test_offline.py` | 10 tests con un LLM "guionado": el grafo, las herramientas, la base y el checkpointer son los reales. |

## Cómo cumple cada criterio

| Criterio | Implementación |
|---|---|
| **Autonomía** | El LLM recibe las herramientas con `llm.bind_tools(TOOLS)` y decide si las usa leyendo sus docstrings. No hay `if/else` que elija la herramienta: el ruteo lo hace `tools_condition` según si el modelo pidió una herramienta o no. |
| **Ciclo de retorno** | Las herramientas devuelven los errores como datos (`{"error": ..., "sugerencia": ...}`) y el prompt indica leer la sugerencia y reintentar. Si una herramienta lanza una excepción (por ejemplo, un argumento de tipo incorrecto), `ToolNode(handle_tool_errors=True)` se la devuelve al modelo como mensaje en lugar de cortar la ejecución. Si después de reintentar falta información, el agente pide una aclaración. |
| **Resiliencia de estado** | `AsyncSqliteSaver` guarda el estado de cada conversación en `checkpoints.sqlite`, indexado por `thread_id`. Con el mismo `thread_id`, el agente recuerda lo anterior, aunque el programa se haya cerrado. Con otro `thread_id`, empieza de cero. |
| **Código limpio** | Python 3.12, *type hints* en todo el código, herramientas y consultas a la base asíncronas (`async def` + `aiosqlite`), y el grafo se ejecuta con `ainvoke`/`astream`. |

### Decisiones de diseño

- **Docstrings como contrato:** cada herramienta explica qué hace, cuándo usarla, qué devuelve y qué hacer si falla. Por ejemplo, `buscar_cliente` aclara que con varios resultados hay que preguntar, y que sin resultados conviene reintentar solo con el apellido.
- **`recursion_limit = 10`:** si el modelo entra en un bucle de herramientas, LangGraph corta en el paso 10. `runner.py` captura el `GraphRecursionError` y devuelve un mensaje controlado.
- **Estado acumulado bajo control:** el checkpointer guarda todo el historial, pero al modelo solo se le envían los últimos 20 mensajes (`trim_messages`), cortando siempre en un mensaje del usuario para no separar una llamada a herramienta de su resultado.
- **`AsyncSqliteSaver`:** es la versión asíncrona de `SqliteSaver`, del mismo paquete (`langgraph-checkpoint-sqlite`). Se usa porque todo el agente es asíncrono; con el `SqliteSaver` síncrono, cada lectura y escritura del estado bloquearía el event loop.

## Cómo levantar el entorno

Requiere **Python 3.12+**.

```bash
cd unidad-05/pre-entrega-5
python -m venv .venv
.venv\Scripts\activate          # Windows  (en Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # completá ANTHROPIC_API_KEY (o configurá OpenAI)
```

```bash
python demo.py                          # prueba con los 4 escenarios → trazas/traza_ejemplo.json y .log
python main.py --thread mi-sesion       # chat interactivo con memoria
python main.py --thread mi-sesion --verbose   # muestra la traza ReAct de cada turno
pytest -v                               # tests offline (sin API keys)
```

La base `erp_demo.db` se crea sola la primera vez.

## Escenarios de la prueba (`demo.py`)

| Turno | Thread | Mensaje | Qué demuestra |
|---|---|---|---|
| 1 | `clientes-…` | ¿Cuántos pedidos tuvo Martina Gómez y cuál fue el total? | Multi-paso: `buscar_cliente` → `buscar_pedidos` (2 herramientas) |
| 2 | `clientes-…` | ¿Y el último? | Memoria: usa el `ultimo_pedido_id` del turno anterior → `detalle_pedido` |
| 3 | `reintento-…` | Necesito saber cuánto compró Martín Gómez. | Ciclo de retorno: sin resultados → reintento por apellido → 2 candidatos → pide aclaración |
| 4 | `reintento-…` | El de Gonnet. | Retoma con la aclaración del usuario |
| 5 | `clientes-…` | Retomando lo de antes: ¿de qué cliente estábamos hablando…? | Persistencia: checkpointer nuevo (como reiniciar el programa) y el mismo `thread_id` |

## Traza de ejecución

La traza completa está en [`trazas/traza_ejemplo.json`](./trazas/traza_ejemplo.json) (estructurada) y [`trazas/traza_ejemplo.log`](./trazas/traza_ejemplo.log) (legible).

Resultado de la corrida real (Anthropic `claude-haiku-5-5`, `recursion_limit=10`):

| Turno | Herramientas usadas | Resultado |
|---|---|---|
| 1 | `buscar_cliente` → `buscar_pedidos` | 3 pedidos, total $14.500 |
| 2 | `detalle_pedido(5042)` | Resolvió "el último" con el contexto del turno 1 |
| 3 | `buscar_cliente("Martín Gómez")` → error → `buscar_cliente("Gómez")` | Reintentó según la sugerencia y, con 2 candidatos, pidió aclaración |
| 4 | `buscar_pedidos(103)` | Joaquín Gómez (Gonnet): $980 |
| 5 | ninguna | Tras reabrir el checkpointer, recordó a Martina Gómez y su pedido 5042 desde SQLite |

Fragmento del turno 3 (ciclo de retorno):

```
[thread reintento-bb5b3e] Usuario: "Necesito saber cuánto compró Martín Gómez."
  → El agente decide usar la herramienta: buscar_cliente(nombre='Martín Gómez')
  → La herramienta devuelve: {"clientes": [], "error": "No hay clientes que coincidan con 'Martín Gómez'.", "sugerencia": "Reintentá buscando solo por el apellido o revisá la ortografía."}
  → El agente decide usar la herramienta: buscar_cliente(nombre='Gómez')
  → La herramienta devuelve: {"clientes": [{"cliente_id": 102, "nombre": "Martina Gómez", ...}, {"cliente_id": 103, "nombre": "Joaquín Gómez", ...}], "cantidad": 2}
  Respuesta: "No encontré un cliente llamado exactamente "Martín Gómez", pero hay dos Gómez en el ERP: ... ¿Cuál de los dos es el que buscás?"
  (llamadas a herramientas en este turno: 2)
```
