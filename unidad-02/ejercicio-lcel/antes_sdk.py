"""
ANTES — versión imperativa con el SDK crudo de Anthropic (estilo Módulo 1).

Se deja solo como referencia para comparar con main.py (versión LCEL).
Fijate todo lo que hay que hacer a mano: formatear el prompt con f-strings,
armar la lista de mensajes, llamar al cliente y extraer el texto del objeto
de respuesta.
"""

import asyncio
import os

from anthropic import AsyncAnthropic
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


async def responder(pregunta: str, nivel: str) -> str:
    client = AsyncAnthropic()
    system = f"Sos un tutor de programación. Explicás para un nivel {nivel}."  # f-string manual
    response = await client.messages.create(
        model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5"),
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": pregunta}],
    )
    # Parsing manual: hay que recorrer los bloques de contenido para sacar el texto
    return "".join(block.text for block in response.content if block.type == "text")


async def main() -> None:
    print(await responder("¿Qué es una corrutina en Python?", "principiante"))


if __name__ == "__main__":
    asyncio.run(main())
