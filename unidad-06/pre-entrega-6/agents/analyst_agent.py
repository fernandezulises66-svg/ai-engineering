"""
Agente de Análisis/Cómputo: toma los datos que encontró el investigador y hace los cálculos
(fechas, porcentajes, montos) con herramientas deterministas, sin "hacer cuentas de cabeza".
"""

import ast
import json
import logging
import operator
from datetime import date, timedelta
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from agents.common import correr_especialista, pregunta_original
from state import OrchestratorState

logger = logging.getLogger("analista")

PROMPT = """Sos el Agente de Análisis del ERP de OLA Muebles. Procesás datos que ya encontró el
Agente de Investigación: no buscás información nueva.

- Hacé TODOS los cálculos con tus herramientas (calcular, dias_entre_fechas, sumar_dias,
  aplicar_porcentaje). Nunca hagas cuentas mentalmente.
- Usá solo las reglas que figuran en los hallazgos del investigador. Si falta un dato para
  calcular, decilo explícitamente ("Falta el dato: ...") en vez de suponerlo.
- Respondé con los RESULTADOS: cada uno con el valor, la cuenta que lo justifica y la regla usada.
  Montos en pesos con separador de miles (ej.: $967.600); fechas en formato dd/mm/aaaa."""

_OPERADORES = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
               ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


def _evaluar(nodo: ast.AST) -> float:
    """Evalúa solo aritmética (números y + - * / **): nada de nombres, llamadas ni atributos."""
    if isinstance(nodo, ast.Constant) and isinstance(nodo.value, (int, float)):
        return nodo.value
    if isinstance(nodo, ast.BinOp) and type(nodo.op) in _OPERADORES:
        return _OPERADORES[type(nodo.op)](_evaluar(nodo.left), _evaluar(nodo.right))
    if isinstance(nodo, ast.UnaryOp) and type(nodo.op) in _OPERADORES:
        return _OPERADORES[type(nodo.op)](_evaluar(nodo.operand))
    raise ValueError("solo se admiten números y los operadores + - * / ** con paréntesis")


def _fecha(texto: str) -> date:
    for sep in ("-", "/"):
        partes = texto.strip().split(sep)
        if len(partes) == 3:
            a, b, c = (int(p) for p in partes)
            return date(a, b, c) if a > 31 else date(c, b, a)   # aaaa-mm-dd o dd/mm/aaaa
    raise ValueError(f"fecha inválida: {texto!r} (usá aaaa-mm-dd o dd/mm/aaaa)")


def _ok(**datos: Any) -> str:
    return json.dumps(datos, ensure_ascii=False)


def _error(mensaje: str) -> str:
    return json.dumps({"error": mensaje}, ensure_ascii=False)


@tool
def calcular(expresion: str) -> str:
    """Evalúa una expresión aritmética y devuelve el resultado exacto.
    Admite números, + - * / ** y paréntesis (ej.: "1180000 * (1 - 0.18)" o "(4200 + 3800) / 2").
    No usa separadores de miles: escribí 1180000, no 1.180.000."""
    try:
        resultado = _evaluar(ast.parse(expresion, mode="eval").body)
    except (SyntaxError, ValueError, ZeroDivisionError) as e:
        return _error(f"No pude evaluar {expresion!r}: {e}")
    return _ok(expresion=expresion, resultado=round(resultado, 2))


@tool
def dias_entre_fechas(desde: str, hasta: str) -> str:
    """Cuenta los días corridos entre dos fechas (aaaa-mm-dd o dd/mm/aaaa).
    Úsala para saber cuántos días pasaron, por ejemplo desde que se emitió un presupuesto hasta hoy.
    Devuelve un número positivo si `hasta` es posterior a `desde`."""
    try:
        d1, d2 = _fecha(desde), _fecha(hasta)
    except ValueError as e:
        return _error(str(e))
    return _ok(desde=d1.isoformat(), hasta=d2.isoformat(), dias_corridos=(d2 - d1).days)


@tool
def sumar_dias(fecha: str, dias: int) -> str:
    """Suma (o resta, si `dias` es negativo) días corridos a una fecha (aaaa-mm-dd o dd/mm/aaaa).
    Úsala para calcular vencimientos, por ejemplo la fecha en que vence un presupuesto."""
    try:
        resultado = _fecha(fecha) + timedelta(days=dias)
    except ValueError as e:
        return _error(str(e))
    return _ok(fecha=fecha, dias=dias, resultado=resultado.isoformat())


@tool
def aplicar_porcentaje(monto: float, porcentaje: float,
                       operacion: Literal["descuento", "recargo", "porcion"]) -> str:
    """Aplica un porcentaje a un monto en pesos.
    - "descuento": monto con el descuento aplicado (1000, 10 → 900)
    - "recargo":   monto con el recargo aplicado (1000, 10 → 1100)
    - "porcion":   qué parte del monto representa ese porcentaje (1000, 50 → 500), ej. una seña."""
    factor = {"descuento": 1 - porcentaje / 100, "recargo": 1 + porcentaje / 100, "porcion": porcentaje / 100}[operacion]
    return _ok(monto=monto, porcentaje=porcentaje, operacion=operacion, resultado=round(monto * factor, 2))


HERRAMIENTAS = [calcular, dias_entre_fechas, sumar_dias, aplicar_porcentaje]


def crear_nodo(llm: BaseChatModel):
    agente = create_react_agent(llm, HERRAMIENTAS, prompt=PROMPT, name="analista")

    async def analista(state: OrchestratorState) -> dict[str, Any]:
        ronda = state.get("delegaciones", 0)
        # Contexto acotado: el encargo + la pregunta + lo último que aportó el investigador.
        hallazgos = [c for c in state.get("contribuciones", []) if c["agente"] == "investigador"]
        datos = hallazgos[-1]["resultado"] if hallazgos else "(el investigador todavía no aportó datos)"
        encargo = (f"Encargo del supervisor: {state['instruccion']}\n\n"
                   f"Pregunta original del usuario: {pregunta_original(state)}\n\n"
                   f"Hallazgos del Agente de Investigación:\n{datos}")
        respuesta, herramientas, _ = await correr_especialista(agente, encargo)
        logger.info("ronda %d · herramientas: %s", ronda, herramientas)
        return {
            "contribuciones": [{"agente": "analista", "ronda": ronda, "instruccion": state["instruccion"],
                                "resultado": respuesta, "herramientas": herramientas, "fuentes": []}],
            "messages": [AIMessage(respuesta, name="analista")],
        }

    return analista
