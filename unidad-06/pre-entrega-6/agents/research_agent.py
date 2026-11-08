"""
Agente de Investigación: busca en el manual técnico del ERP (Pinecone de la pre-entrega 4)
las reglas de negocio y los datos que hacen falta. No calcula: solo encuentra y cita.
"""

import json
import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import create_react_agent

import config
from agents.common import correr_especialista, pregunta_original
from knowledge import KnowledgeBase
from state import OrchestratorState

logger = logging.getLogger("investigador")

PROMPT = """Sos el Agente de Investigación del ERP de OLA Muebles. Tu única fuente es el manual técnico
del ERP, que consultás con la herramienta `buscar_manual`.

- Buscá los datos que te pide el encargo: una búsqueda por tema, y como máximo 5 en total. Si una
  búsqueda no trae lo que necesitás, reformulá la consulta con otras palabras.
- Leé el texto COMPLETO de cada fragmento: una regla puede estar al final, dentro de una tabla o de una lista.
- No hagas cálculos ni saques conclusiones numéricas: eso lo hace el Agente de Análisis.
- Respondé breve (como máximo 10 líneas) con una lista de HALLAZGOS: el dato textual y su fuente entre corchetes,
  por ejemplo: "- Un presupuesto vale 15 días corridos [03-api-ventas.md]".
- Si algo no figura en el manual, decilo explícitamente ("No encontrado en el manual: ...")."""


def crear_herramientas(kb: KnowledgeBase) -> list[BaseTool]:
    @tool
    async def buscar_manual(consulta: str) -> str:
        """Busca en el manual técnico del ERP de OLA Muebles (ventas, presupuestos, roles y permisos,
        caja, inventario, integraciones con AFIP y Tiendanube, códigos de error).

        Usala para encontrar reglas de negocio: plazos, porcentajes, qué rol puede hacer qué,
        estados de una venta o el significado de un código de error. Escribí la consulta como una
        frase corta sobre UN tema (ej.: "validez de un presupuesto", "seña de muebles a medida").

        Devuelve un JSON con los fragmentos más relevantes: fuente (archivo del manual), título y texto.
        Si no hay resultados, devuelve un `error` con una sugerencia para reformular.
        """
        fragmentos = await kb.buscar(consulta)
        if not fragmentos:
            return json.dumps({"fragmentos": [], "error": f"Sin resultados para '{consulta}'.",
                               "sugerencia": "Probá con sinónimos o con una consulta más general."},
                              ensure_ascii=False)
        # Texto completo del chunk: un recorte corto pierde las reglas que están al final del fragmento
        recortados = [{**f, "texto": f["texto"][:config.FRAGMENTO_MAX_CHARS]} for f in fragmentos]
        return json.dumps({"fragmentos": recortados}, ensure_ascii=False)

    return [buscar_manual]


def crear_nodo(llm: BaseChatModel, kb: KnowledgeBase):
    agente = create_react_agent(llm, crear_herramientas(kb), prompt=PROMPT, name="investigador")

    async def investigador(state: OrchestratorState) -> dict[str, Any]:
        ronda = state.get("delegaciones", 0)
        # Contexto acotado: el encargo del supervisor + la pregunta del usuario. Nada más.
        encargo = f"Encargo del supervisor: {state['instruccion']}\n\nPregunta original del usuario: {pregunta_original(state)}"
        respuesta, herramientas, resultados = await correr_especialista(agente, encargo)
        fuentes = sorted({f["fuente"] for r in resultados if isinstance(r, dict) for f in r.get("fragmentos", [])})
        logger.info("ronda %d · %d búsquedas · fuentes: %s", ronda, len(herramientas), fuentes)
        return {
            "contribuciones": [{"agente": "investigador", "ronda": ronda, "instruccion": state["instruccion"],
                                "resultado": respuesta, "herramientas": herramientas, "fuentes": fuentes}],
            "messages": [AIMessage(respuesta, name="investigador")],
        }

    return investigador
