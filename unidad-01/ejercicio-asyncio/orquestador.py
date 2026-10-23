"""
Unidad 1 · Ejercicio: Orquestador concurrente de modelos (Python 3.12+)

Ejecutar:  python3.12 unidad1_orquestador.py

Escenarios:
  1. gather de 3 modelos con timeout de 2s por llamada  -> los que responden se conservan.
  2. gather de 3 modelos con timeout de 2s a la ejecución TOTAL -> se captura el
     TimeoutError y el programa sigue (pero se pierden los resultados parciales).
  3. 10 llamadas disparadas a la vez, limitadas a 2 concurrentes con Semaphore.
"""

import asyncio
import logging
import random
import time

T0 = time.perf_counter()


class RelativeTimeFormatter(logging.Formatter):
    """Muestra el tiempo transcurrido desde el arranque, para ver la concurrencia."""

    def format(self, record: logging.LogRecord) -> str:
        record.elapsed = f"{time.perf_counter() - T0:6.2f}s"
        return super().format(record)


handler = logging.StreamHandler()
handler.setFormatter(RelativeTimeFormatter("[%(elapsed)s] %(message)s"))
log = logging.getLogger("orquestador")
log.addHandler(handler)
log.setLevel(logging.INFO)


# --- 1. Simulación de llamadas a modelos (I/O-bound) ---------------------------

async def _fake_llm(model: str, prompt: str, latency: float) -> dict:
    log.info(f"→ {model:<12} inicia  (latencia {latency:.1f}s)")
    await asyncio.sleep(latency)  # await, NUNCA time.sleep: no bloquea el event loop
    log.info(f"← {model:<12} responde")
    return {"model": model, "ok": True, "answer": f"[{model}] respuesta a '{prompt}'"}


async def gpt_4_call(prompt: str) -> dict:
    return await _fake_llm("gpt-4", prompt, 0.8)


async def claude_3_call(prompt: str) -> dict:
    return await _fake_llm("claude-3", prompt, 1.3)


async def local_llama_call(prompt: str) -> dict:
    return await _fake_llm("local-llama", prompt, 2.6)  # a propósito > 2s


MODELS = {
    "gpt-4": gpt_4_call,
    "claude-3": claude_3_call,
    "local-llama": local_llama_call,
}


# --- 2. Resiliencia: timeout por llamada ---------------------------------------

async def with_timeout(name: str, coro, seconds: float = 2.0) -> dict:
    try:
        async with asyncio.timeout(seconds):
            return await coro
    except TimeoutError:
        log.warning(f"✗ {name:<12} TIMEOUT (> {seconds}s) — se aplica fallback")
        return {"model": name, "ok": False, "error": "timeout"}


async def escenario_1(prompt: str) -> None:
    log.info("=== Escenario 1: gather + timeout de 2s por llamada ===")
    results = await asyncio.gather(
        *(with_timeout(name, fn(prompt)) for name, fn in MODELS.items()),
        return_exceptions=True,  # un error inesperado no tumba al resto
    )
    for r in results:
        if isinstance(r, Exception):
            log.error(f"  excepción inesperada: {r!r}")
        elif r["ok"]:
            log.info(f"  OK    {r['model']}")
        else:
            log.info(f"  FALLO {r['model']} -> {r['error']}")


# --- 3. Resiliencia: timeout a la ejecución total ------------------------------

async def escenario_2(prompt: str) -> None:
    log.info("=== Escenario 2: timeout de 2s a la ejecución TOTAL ===")
    try:
        async with asyncio.timeout(2.0):
            results = await asyncio.gather(
                *(fn(prompt) for fn in MODELS.values()),
                return_exceptions=True,
            )
            log.info(f"  Todas respondieron: {len(results)}")
    except TimeoutError:
        # asyncio.timeout cancela las tareas pendientes y convierte el
        # CancelledError interno en TimeoutError al salir del bloque.
        log.error("  ✗ El lote completo superó 2s. Se cancelaron las pendientes; el programa sigue.")


# --- 4. Control de flujo: Semaphore(2) con 10 llamadas -------------------------

running = 0  # contador para demostrar que nunca hay más de 2 en vuelo


async def throttled_call(i: int, sem: asyncio.Semaphore) -> dict:
    global running
    log.info(f"  · req-{i:02d} encolada")
    async with sem:
        running += 1
        log.info(f"  ▶ req-{i:02d} ejecutando (en vuelo: {running})")
        try:
            async with asyncio.timeout(2.0):
                await asyncio.sleep(random.uniform(0.3, 1.0))
                return {"id": i, "ok": True}
        except TimeoutError:
            return {"id": i, "ok": False, "error": "timeout"}
        finally:
            running -= 1


async def escenario_3() -> None:
    log.info("=== Escenario 3: 10 llamadas, máximo 2 concurrentes ===")
    sem = asyncio.Semaphore(2)
    start = time.perf_counter()
    results = await asyncio.gather(
        *(throttled_call(i, sem) for i in range(10)),
        return_exceptions=True,
    )
    ok = sum(1 for r in results if isinstance(r, dict) and r["ok"])
    log.info(f"  {ok}/10 OK en {time.perf_counter() - start:.2f}s")


# --- Punto de entrada ----------------------------------------------------------

async def main() -> None:
    prompt = "¿Qué es asyncio?"
    await escenario_1(prompt)
    await escenario_2(prompt)
    await escenario_3()


if __name__ == "__main__":
    asyncio.run(main())
