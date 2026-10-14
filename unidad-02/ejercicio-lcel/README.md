# Ejercicio · Refactorización a LCEL asíncrono

Migración de una llamada imperativa al SDK de Anthropic (`antes_sdk.py`) a una cadena declarativa con LCEL (`main.py`).

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "Sos un tutor de programación. Explicás para un nivel {nivel}, ..."),
    ("human", "{pregunta}"),
])
chain = prompt | ChatAnthropic(model="claude-haiku-5-5", max_tokens=512) | StrOutputParser()

respuesta = await chain.ainvoke({"pregunta": "¿Qué es una corrutina en Python?", "nivel": "principiante"})
```

## Antes vs. después

| | `antes_sdk.py` (SDK crudo) | `main.py` (LCEL) |
|---|---|---|
| Prompt | f-string a mano | `ChatPromptTemplate` con roles system/human |
| Llamada | `client.messages.create(...)` | parte de la cadena (`\|`) |
| Parsing | recorrer `response.content` a mano | `StrOutputParser()` |
| Async, batch, streaming | hay que implementarlo | `ainvoke`, `abatch`, `astream` incluidos |

## Criterios de aceptación

- [x] El flujo principal usa solo LCEL: `prompt | model | parser`.
- [x] La ejecución es asíncrona: `await chain.ainvoke({...})`.
- [x] La salida es texto plano gracias a `StrOutputParser`, no un `AIMessage`.
- [x] El `ChatPromptTemplate` tiene roles System y Human, y las variables (`{pregunta}`, `{nivel}`) coinciden con las claves del diccionario.

> **Sobre `temperature`:** la consigna pide configurarla, pero los modelos Claude 5.x devuelven error 400 si se envía. Por eso se configuran `model`, `max_tokens`, `max_retries` y `timeout`, y `temperature` solo se envía si se define `ANTHROPIC_TEMPERATURE` en el `.env` (para modelos anteriores).

## Cómo correrlo

Desde la carpeta `unidad-02`, con el `.env` configurado (ver [pre-entrega-2](../pre-entrega-2/README.md#cómo-correrlo)):

```bash
python ejercicio-lcel/main.py
```
