"""
Unidad 2 · Ejercicio: Refactorización a LCEL asíncrono.

Cadena:  ChatPromptTemplate (system + human) | ChatAnthropic | StrOutputParser
Entrada: {"pregunta": "...", "nivel": "..."}
Salida:  str (texto plano, no un AIMessage)

Ejecutar desde la carpeta unidad-02:
    python ejercicio-lcel/main.py
"""

import asyncio
import os

from dotenv import find_dotenv, load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(find_dotenv())


# --- 1. Modelo: reemplaza al cliente manual del SDK --------------------------
def build_model() -> ChatAnthropic:
    params = {
        "model": os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5"),
        "max_tokens": 512,
        "max_retries": 2,  # reintentos de red/429 gestionados por LangChain
        "timeout": 30,
    }
    # Los modelos Claude 5.x rechazan temperature (error 400). Solo se envía
    # si se configuró explícitamente para un modelo que la acepte.
    if temperature := os.getenv("ANTHROPIC_TEMPERATURE"):
        params["temperature"] = float(temperature)
    return ChatAnthropic(**params)


# --- 2. Prompt con roles definidos (System / Human) --------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Sos un tutor de programación. Explicás para un nivel {nivel}, "
            "en español rioplatense, en no más de 5 oraciones.",
        ),
        ("human", "{pregunta}"),
    ]
)

# --- 3. Cadena LCEL: prompt | modelo | parser --------------------------------
chain = prompt | build_model() | StrOutputParser()


# --- 4. Ejecución asíncrona --------------------------------------------------
async def main() -> None:
    entrada = {"pregunta": "¿Qué es una corrutina en Python?", "nivel": "principiante"}
    respuesta = await chain.ainvoke(entrada)

    # StrOutputParser devuelve texto plano (una subclase de str), no un AIMessage
    print(f"¿Es texto plano? {isinstance(respuesta, str)}")
    print("-" * 60)
    print(respuesta)


if __name__ == "__main__":
    asyncio.run(main())
