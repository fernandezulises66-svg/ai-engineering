"""Ejecuta el orquestador y arma una traza legible del flujo de delegación."""

import json
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from langgraph.graph.state import CompiledStateGraph

import config
from graph import estado_inicial, run_config

PREGUNTA_DEMO = ("Hoy es 10/11/2026. Un cliente pidió el 28/10/2026 un presupuesto por una biblioteca a medida "
                 "de $1.180.000 y quiere confirmarlo con un 18 % de descuento. Según el manual del ERP: "
                 "¿el presupuesto sigue vigente?, ¿cuánto queda el precio con el descuento y cuánto tiene que "
                 "pagar de seña?, ¿qué rol puede aprobar ese descuento?")


def _corto(texto: str, n: int = 600) -> str:
    texto = texto.strip()
    return texto if len(texto) <= n else texto[:n] + " […]"


def describir(nodo: str, update: dict[str, Any]) -> str:
    """Una línea (o bloque) por paso del grafo."""
    if nodo == "supervisor":
        d = update["decisiones"][-1]
        instr = f"\n    instrucción: {update['instruccion']}" if update.get("instruccion") else ""
        return f"[SUPERVISOR · ronda {d['ronda']}] → {d['siguiente']} ({d['origen']}): {d['motivo']}{instr}"
    if nodo in ("investigador", "analista"):
        c = update["contribuciones"][-1]
        fuentes = f" · fuentes: {', '.join(c['fuentes'])}" if c["fuentes"] else ""
        return (f"[{nodo.upper()}] herramientas: {', '.join(c['herramientas']) or 'ninguna'}{fuentes}\n"
                + "\n".join("    " + linea for linea in _corto(c["resultado"]).splitlines()))
    if nodo == "sintetizador":
        return "[SINTETIZADOR] respuesta final:\n" + "\n".join("    " + l for l in update["respuesta_final"].splitlines())
    return f"[{nodo}]"


async def ejecutar(graph: CompiledStateGraph, pregunta: str,
                   mostrar: Callable[[str], None] | None = print) -> dict[str, Any]:
    """Corre el grafo paso a paso (stream de actualizaciones) y devuelve la traza completa."""
    pasos: list[dict[str, Any]] = []
    inicio = time.perf_counter()
    estado: dict[str, Any] = estado_inicial(pregunta)
    async for evento in graph.astream(estado, run_config(), stream_mode="updates"):
        for nodo, update in evento.items():
            update = update or {}
            pasos.append({"nodo": nodo, **{k: v for k, v in update.items() if k != "messages"}})
            for clave in ("contribuciones", "decisiones"):
                estado[clave] = estado.get(clave, []) + update.get(clave, [])
            estado.update({k: v for k, v in update.items() if k not in ("contribuciones", "decisiones", "messages")})
            if mostrar:
                mostrar(describir(nodo, update) + "\n")
    return {
        "generada": datetime.now().isoformat(timespec="seconds"),
        "proveedor": config.LLM_PROVIDER,
        "base_de_conocimiento": config.RESEARCH_BACKEND,
        "pregunta": pregunta,
        "duracion_segundos": round(time.perf_counter() - inicio, 1),
        "recorrido": ["supervisor" if p["nodo"] == "supervisor" else p["nodo"] for p in pasos],
        "delegaciones": estado.get("delegaciones", 0),
        "task_completed": estado.get("task_completed", False),
        "respuesta_final": estado.get("respuesta_final", ""),
        "diagnostico_sintesis": estado.get("diagnostico_sintesis", {}),
        "decisiones": estado.get("decisiones", []),
        "contribuciones": estado.get("contribuciones", []),
        "pasos": pasos,
    }


def guardar(traza: dict[str, Any], nombre: str = "traza_ejemplo") -> None:
    config.TRAZAS_DIR.mkdir(exist_ok=True)
    (config.TRAZAS_DIR / f"{nombre}.json").write_text(json.dumps(traza, ensure_ascii=False, indent=2), encoding="utf-8")
    log = [f"Pregunta: {traza['pregunta']}", f"Recorrido: {' → '.join(traza['recorrido'])} → END", ""]
    log += [describir(p["nodo"], p) for p in traza["pasos"]]
    (config.TRAZAS_DIR / f"{nombre}.log").write_text("\n\n".join(log) + "\n", encoding="utf-8")
