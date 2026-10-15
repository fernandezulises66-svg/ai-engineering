"""
Cadena RAG asíncrona (LCEL) sobre la base vectorial de OLA Muebles.

    {"pregunta"} ─► recuperar (embedding + top_k en Chroma) ─► formatear contexto
                 ─► prompt grounded ─► ChatAnthropic ─► PydanticOutputParser
                 ─► armar_respuesta (mapea las citas a fragmentos reales) ─► RAGResponse
"""

import asyncio
import logging
from operator import itemgetter

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough
from pydantic import BaseModel, Field

import config

logger = logging.getLogger("rag")

NO_LO_SE = "No lo sé: esa información no está en la documentación disponible."


# --- Esquemas -------------------------------------------------------------------
class RespuestaLLM(BaseModel):
    """Lo que el modelo debe devolver (se parsea con PydanticOutputParser)."""

    respuesta: str = Field(description="Respuesta en español, basada solo en el CONTEXTO.")
    encontrada_en_contexto: bool = Field(
        description="true si el CONTEXTO contiene la información para responder; false en caso contrario."
    )
    fragmentos_usados: list[int] = Field(
        default_factory=list,
        description="Números de los fragmentos del CONTEXTO usados para responder, por ejemplo [1, 3]. Vacío si no se encontró.",
    )


class Referencia(BaseModel):
    fuente: str
    titulo: str
    chunk: int
    score: float | None = Field(default=None, description="Similitud coseno con la pregunta (0 a 1).")
    extracto: str


class RAGResponse(BaseModel):
    """Respuesta final del sistema: texto + referencias verificables."""

    pregunta: str
    respuesta: str
    encontrada_en_contexto: bool
    referencias: list[Referencia]


parser = PydanticOutputParser(pydantic_object=RespuestaLLM)

# --- Prompt: "filtro de veracidad" ----------------------------------------------
SYSTEM_PROMPT = """Sos el asistente de atención al cliente y del equipo interno de OLA Muebles.

Reglas obligatorias:
1. Respondé ÚNICAMENTE con la información del CONTEXTO. No uses conocimiento propio ni supongas datos.
2. Si el CONTEXTO no contiene la respuesta, respondé exactamente: "{no_lo_se}", con encontrada_en_contexto=false y fragmentos_usados vacío.
3. Si el CONTEXTO responde solo una parte, respondé esa parte y aclarás qué parte no figura en la documentación.
4. Citá en fragmentos_usados los números de los fragmentos que usaste.
5. Respondé en español rioplatense, de forma breve y concreta (máximo 5 oraciones)."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "CONTEXTO:\n{contexto}\n\nPREGUNTA: {pregunta}\n\n{format_instructions}"),
    ]
).partial(no_lo_se=NO_LO_SE, format_instructions=parser.get_format_instructions())


# --- Pasos de la cadena ---------------------------------------------------------
def format_docs(docs: list[Document]) -> str:
    if not docs:
        return "(sin fragmentos relevantes)"
    return "\n\n".join(
        f"[{i}] Fuente: {d.metadata['source']} (fragmento {d.metadata.get('chunk', '?')})\n{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )


def armar_respuesta(datos: dict) -> RAGResponse:
    """Convierte las citas del modelo (números) en referencias a fragmentos reales.

    Las referencias salen de los documentos recuperados, no del texto del modelo,
    así que no puede "inventar" una fuente.
    """
    docs: list[Document] = datos["docs"]
    salida: RespuestaLLM = datos["salida"]

    referencias = []
    if salida.encontrada_en_contexto:
        for n in dict.fromkeys(salida.fragmentos_usados):  # sin duplicados, en orden
            if not 1 <= n <= len(docs):
                logger.warning("El modelo citó el fragmento [%d], que no existe: se ignora", n)
                continue
            d = docs[n - 1]
            referencias.append(Referencia(
                fuente=d.metadata["source"],
                titulo=d.metadata.get("title", d.metadata["source"]),
                chunk=d.metadata.get("chunk", 0),
                score=d.metadata.get("score"),
                extracto=" ".join(d.page_content.split())[:240] + "…",
            ))
        if not referencias:
            logger.warning("Respuesta marcada como encontrada pero sin fragmentos citados válidos")

    logger.info("Respuesta generada (encontrada_en_contexto=%s, %d referencias)",
                salida.encontrada_en_contexto, len(referencias))
    return RAGResponse(
        pregunta=datos["pregunta"],
        respuesta=salida.respuesta if salida.encontrada_en_contexto else NO_LO_SE,
        encontrada_en_contexto=salida.encontrada_en_contexto,
        referencias=referencias,
    )


def build_rag_chain(store: Chroma, llm: BaseChatModel | None = None, k: int = config.TOP_K) -> Runnable:
    llm = llm or config.get_llm()

    async def recuperar(pregunta: str) -> list[Document]:
        # La pregunta se convierte en embedding con el MISMO modelo que se usó al indexar
        resultados = await store.asimilarity_search_with_relevance_scores(pregunta, k=k)
        docs = []
        for doc, score in resultados:
            doc.metadata["score"] = round(score, 3)
            docs.append(doc)
        logger.info("Recuperados %d fragmentos: %s", len(docs),
                    ", ".join(f"{d.metadata['chunk_id']} ({d.metadata['score']})" for d in docs))
        return docs

    generacion = (prompt | llm | parser).with_retry(
        retry_if_exception_type=(OutputParserException,),
        stop_after_attempt=3,
        wait_exponential_jitter=True,
    )

    return (
        RunnableParallel(docs=itemgetter("pregunta") | RunnableLambda(recuperar), pregunta=itemgetter("pregunta"))
        | RunnablePassthrough.assign(contexto=lambda x: format_docs(x["docs"]))
        | RunnablePassthrough.assign(salida=generacion)
        | RunnableLambda(armar_respuesta)
    )


# --- API pública ----------------------------------------------------------------
_chain: Runnable | None = None


def get_chain() -> Runnable:
    """Construye la cadena una sola vez. ingest() reutiliza la base si ya está al día."""
    global _chain
    if _chain is None:
        from ingest import ingest

        _chain = build_rag_chain(ingest(force=False))
    return _chain


async def get_rag_response(query: str, chain: Runnable | None = None) -> RAGResponse:
    if not query or not query.strip():
        raise ValueError("la pregunta está vacía")
    logger.info("Pregunta: %s", query)
    return await (chain or get_chain()).ainvoke({"pregunta": query.strip()})


if __name__ == "__main__":
    import sys

    pregunta = " ".join(sys.argv[1:]) or "¿Cuánto cuesta el envío a CABA?"
    print(asyncio.run(get_rag_response(pregunta)).model_dump_json(indent=2))
