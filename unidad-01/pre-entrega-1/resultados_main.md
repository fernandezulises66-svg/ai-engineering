# Resultados reales de `main.py --both`

Ejecución del 24 de octubre de 2026 en Windows (Python 3.13), con `gpt-6-luna` y `claude-haiku-5-5`.

## Salida del programa

```text
======================================================================
Proveedor: openai · modelo: gpt-6-luna · max_tokens: 1024

--- 1. Modo normal (generate) ---
La entropía mide cuántas formas distintas puede organizarse un sistema sin que cambie su estado observable; en teoría de la información, mide la incertidumbre de un mensaje. En termodinámica, suele aumentar en los procesos espontáneos, aunque decir que es simplemente «desorden» es una simplificación.
[68.60s · tokens: input_tokens=31 output_tokens=119 · fin: stop]

--- 2. Streaming (stream con yield) ---
La entropía mide cuántas maneras distintas puede organizarse un sistema sin cambiar lo que observamos de él; también puede entenderse como una medida de incertidumbre o de cuán dispersa está la energía. En termodinámica, la segunda ley dice que la entropía total de un sistema aislado no disminuye con el tiempo.
[primer token (TTFT): 8.27s · total: 14.78s]

--- 3. Error controlado (API key inválida) ---
ok=False · tipo=timeout · status=None
→ El programa sigue funcionando: el error se devolvió como dato, no como excepción.

======================================================================
Proveedor: anthropic · modelo: claude-haiku-5-5 · max_tokens: 1024

--- 1. Modo normal (generate) ---
La entropía es una magnitud que mide el grado de desorden o de aleatoriedad de un sistema, y también la cantidad de energía que no puede aprovecharse para producir trabajo. En termodinámica, la segunda ley establece que la entropía de un sistema aislado tiende a aumentar con el tiempo. En teoría de la información, mide la incertidumbre promedio asociada a una fuente de datos.
[12.45s · tokens: input_tokens=39 output_tokens=140 · fin: end_turn]

--- 2. Streaming (stream con yield) ---
La entropía es una magnitud que mide el grado de desorden o de dispersión de la energía en un sistema, y en termodinámica se relaciona con la cantidad de microestados posibles que son compatibles con su estado macroscópico. Según la segunda ley de la termodinámica, la entropía de un sistema aislado tiende a aumentar con el tiempo. En teoría de la información, el término también se usa para cuantificar la incertidumbre promedio de una fuente de datos.
[primer token (TTFT): 4.70s · total: 10.06s]

--- 3. Error controlado (API key inválida) ---
ok=False · tipo=authentication · status=401
→ El programa sigue funcionando: el error se devolvió como dato, no como excepción.
```

## Logs (stderr)

Durante la prueba hubo cortes de red. Los logs muestran los reintentos automáticos en acción:

```text
11:27:33 | WARNING | [openai] connection (intento 1) → reintento en 1.5s
11:27:52 | WARNING | [openai] connection (intento 2) → reintento en 2.4s
11:28:09 | INFO    | [openai] respuesta OK (301 caracteres)
11:28:29 | WARNING | [openai] timeout (intento 1) → reintento en 1.4s
11:28:40 | WARNING | [openai] timeout (intento 2) → reintento en 2.5s
11:28:47 | ERROR   | [openai] error controlado: timeout
11:29:00 | INFO    | [anthropic] respuesta OK (378 caracteres)
11:29:15 | WARNING | [anthropic] timeout (intento 1) → reintento en 1.2s
11:29:21 | WARNING | [anthropic] timeout (intento 2) → reintento en 2.2s
11:29:27 | ERROR   | [anthropic] error controlado: authentication
```

- **OpenAI, modo normal:** 2 errores de conexión, reintentos con backoff de 1,5 s y 2,4 s, y respuesta correcta al tercer intento (de ahí los 68 s).
- **Streaming:** el texto empezó a llegar a los 8,3 s (OpenAI) y 4,7 s (Anthropic), antes de terminar la respuesta completa. En Anthropic, el stream se recuperó después de 2 timeouts al iniciar.
- **Error controlado:** con una key inválida, Anthropic devolvió `authentication` (401). En OpenAI el corte de red se adelantó y devolvió `timeout`. En los dos casos el programa siguió funcionando.
