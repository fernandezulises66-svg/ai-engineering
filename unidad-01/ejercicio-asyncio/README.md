# Unidad 1 · Orquestador concurrente de modelos

Simula un backend que consulta varios modelos de IA en paralelo con `asyncio`.

```bash
python orquestador.py
```

## Qué demuestra

1. **`asyncio.gather`**: las 3 llamadas (`gpt_4_call`, `claude_3_call`, `local_llama_call`) arrancan al mismo tiempo.
2. **Timeout por llamada (2s)**: `local-llama` tarda 2.6s → se captura `TimeoutError` y se conservan las demás respuestas.
3. **Timeout a la ejecución total (2s)**: se captura `TimeoutError` y el programa sigue, pero se pierden los resultados parciales.
4. **`asyncio.Semaphore(2)`**: se disparan 10 llamadas, pero nunca hay más de 2 en vuelo.

No se usa `time.sleep` (bloqueante) en ningún lugar; los logs muestran el tiempo relativo para evidenciar la concurrencia.

## Salida de ejemplo

```
[  0.00s] → gpt-4        inicia  (latencia 0.8s)
[  0.00s] → claude-3     inicia  (latencia 1.3s)
[  0.00s] → local-llama  inicia  (latencia 2.6s)
[  0.80s] ← gpt-4        responde
[  1.30s] ← claude-3     responde
[  2.00s] ✗ local-llama  TIMEOUT (> 2.0s) — se aplica fallback
...
[  4.01s]   ▶ req-00 ejecutando (en vuelo: 1)
[  4.01s]   ▶ req-01 ejecutando (en vuelo: 2)
[  4.38s]   ▶ req-02 ejecutando (en vuelo: 2)
```

## Pregunta de reflexión

> ¿Usarías `asyncio.gather` con 10.000 llamadas de embeddings a la vez?

No. Dispararlas todas juntas provoca errores `429: Too Many Requests`, abre miles de conexiones y consume memoria innecesaria. Lo correcto es encolarlas con `gather` pero limitar la concurrencia con un `Semaphore` (N llamadas simultáneas), idealmente agrupando textos en lotes (las APIs de embeddings aceptan varios por request) y agregando reintentos con backoff exponencial ante 429.
