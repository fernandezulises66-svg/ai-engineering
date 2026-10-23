"""
Script de validación: hace la misma pregunta en modo normal y en streaming,
y muestra que un error (API key inválida) se devuelve controlado, sin crash.

    python main.py                      # usa LLM_PROVIDER del .env (por defecto: anthropic)
    python main.py --provider openai    # fuerza un proveedor
    python main.py --both               # prueba los dos proveedores, uno después del otro
"""

import argparse
import asyncio
import logging
import sys
import time

# La consola de Windows usa cp1252 y no puede imprimir algunos símbolos (✗, —, acentos al redirigir)
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from manager import AsyncLLMManager
from schemas import ChatMessage

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
for ruidoso in ("httpx", "httpx2", "openai", "anthropic"):
    logging.getLogger(ruidoso).setLevel(logging.WARNING)

PREGUNTA = [
    ChatMessage(role="system", content="Respondé en español, en no más de 3 oraciones."),
    ChatMessage(role="user", content="¿Qué es la entropía?"),
]


async def probar(provider: str | None) -> None:
    async with AsyncLLMManager.from_env(provider) as llm:
        cfg = llm.config
        print(f"\n{'=' * 70}\nProveedor: {cfg.provider} · modelo: {cfg.model} · max_tokens: {cfg.max_tokens}")

        # 1. Modo normal
        print("\n--- 1. Modo normal (generate) ---")
        inicio = time.perf_counter()
        r = await llm.generate(PREGUNTA)
        if r.ok:
            print(r.content)
            print(f"[{time.perf_counter() - inicio:.2f}s · tokens: {r.usage} · fin: {r.finish_reason}]")
        else:
            print(f"✗ Error controlado: {r.error.type} — {r.error.message}")

        # 2. Streaming: el texto aparece a medida que llega
        print("\n--- 2. Streaming (stream con yield) ---")
        inicio, primer_token = time.perf_counter(), None
        async for chunk in llm.stream(PREGUNTA):
            if chunk.delta:
                primer_token = primer_token or time.perf_counter() - inicio
                print(chunk.delta, end="", flush=True)
            if chunk.error:
                print(f"\n✗ Error controlado: {chunk.error.type} — {chunk.error.message}")
        ttft = f"{primer_token:.2f}s" if primer_token else "—"
        print(f"\n[primer token (TTFT): {ttft} · total: {time.perf_counter() - inicio:.2f}s]")

    # 3. Error controlado: una key inválida no rompe el programa
    print("\n--- 3. Error controlado (API key inválida) ---")
    from clients import AnthropicClient, OpenAIClient
    from anthropic import AsyncAnthropic
    from openai import AsyncOpenAI

    fake = (OpenAIClient(cfg, AsyncOpenAI(api_key="sk-invalida", max_retries=0)) if cfg.provider == "openai"
            else AnthropicClient(cfg, AsyncAnthropic(api_key="sk-ant-invalida", max_retries=0)))
    async with AsyncLLMManager(cfg, client=fake) as roto:
        r = await roto.generate("Hola")
        print(f"ok={r.ok} · tipo={r.error.type if r.error else None} · status={r.error.status_code if r.error else None}")
        print("→ El programa sigue funcionando: el error se devolvió como dato, no como excepción.")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["openai", "anthropic"])
    parser.add_argument("--both", action="store_true")
    args = parser.parse_args()
    for provider in (["openai", "anthropic"] if args.both else [args.provider]):
        await probar(provider)


if __name__ == "__main__":
    asyncio.run(main())
