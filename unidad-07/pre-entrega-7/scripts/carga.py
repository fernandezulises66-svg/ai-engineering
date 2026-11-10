"""
Prueba de carga: 5 peticiones concurrentes contra la API.

    python scripts/carga.py                 # las críticas quedan en WAITING_APPROVAL
    python scripts/carga.py --aprobar       # además las aprueba como "gerencia" y espera a que terminen

Imprime una tabla por job (estado, recorrido, latencia vista por el cliente) y la p50/p95 del lado del
cliente. Las métricas oficiales (costo por ejecución y latencia p95) se leen en el dashboard de Phoenix.
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PREGUNTAS = [
    "Hoy es 10/11/2026. Un cliente pidió el 28/10/2026 un presupuesto por una biblioteca a medida de $1.180.000 "
    "y quiere confirmarlo con un 18 % de descuento. Según el manual del ERP: ¿el presupuesto sigue vigente?, "
    "¿cuánto queda el precio con el descuento y cuánto tiene que pagar de seña?, ¿qué rol puede aprobar ese descuento?",
    "¿Qué significa el error CAJA-422-JUST y cómo se soluciona?",
    "Hoy es 10/11/2026. Un cierre de caja dio una diferencia de $7.300. Según el manual, ¿hace falta "
    "justificación? ¿Por cuánto supera el umbral?",
    "Hoy es 10/11/2026. Un presupuesto se emitió el 30/10/2026. ¿Cuántos días de validez le quedan y qué error "
    "devuelve la API si se intenta convertir en venta después de vencido?",
    "Hoy es 10/11/2026. Aplicá un descuento del 18 % al presupuesto P-1042 de $1.180.000 (biblioteca a medida, "
    "emitido el 28/10/2026) y registralo en el ERP con la seña que corresponda.",
]
FINALES = {"DONE", "FAILED"}


def percentil(valores: list[float], p: float) -> float:
    ordenados = sorted(valores)
    k = (len(ordenados) - 1) * p
    i, f = int(k), k - int(k)
    return ordenados[i] + (ordenados[min(i + 1, len(ordenados) - 1)] - ordenados[i]) * f


async def correr_job(cliente: httpx.AsyncClient, n: int, pregunta: str, aprobar: bool) -> dict:
    inicio = time.perf_counter()
    r = await cliente.post("/tasks", json={"pregunta": pregunta})
    r.raise_for_status()
    job_id = r.json()["job_id"]
    print(f"[{n}] encolado {job_id} en {(time.perf_counter() - inicio) * 1000:.0f} ms")
    espera_humana = 0.0
    while True:
        await asyncio.sleep(2)
        job = (await cliente.get(f"/tasks/{job_id}")).json()
        if job["status"] in FINALES:
            break
        if job["status"] == "WAITING_APPROVAL":
            if not aprobar:
                break
            print(f"[{n}] WAITING_APPROVAL → apruebo como gerencia")
            t = time.perf_counter()
            await cliente.post(f"/tasks/{job_id}/approve",
                               json={"aprobado": True, "revisor": "gerencia", "comentario": "prueba de carga"})
            espera_humana += time.perf_counter() - t
    total = time.perf_counter() - inicio
    resultado = job.get("resultado") or {}
    return {"n": n, "job_id": job_id, "status": job["status"], "latencia_s": round(total, 1),
            "recorrido": " → ".join(resultado.get("recorrido", [])), "es_critica": resultado.get("es_critica"),
            "accion": resultado.get("accion_resultado", ""), "error": job.get("error", "")}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--aprobar", action="store_true", help="aprobar automáticamente las tareas críticas")
    args = parser.parse_args()

    async with httpx.AsyncClient(base_url=args.url, timeout=30) as cliente:
        inicio = time.perf_counter()
        resultados = await asyncio.gather(*(correr_job(cliente, i + 1, p, args.aprobar)
                                            for i, p in enumerate(PREGUNTAS)))
        total = time.perf_counter() - inicio

    print(f"\n{'#':<3}{'estado':<18}{'latencia':>9}  recorrido")
    for r in resultados:
        print(f"{r['n']:<3}{r['status']:<18}{r['latencia_s']:>8}s  {r['recorrido']}"
              + (f"  [crítica: {r['accion'] or 'esperando aprobación'}]" if r["es_critica"] else "")
              + (f"  ERROR: {r['error']}" if r["error"] else ""))
    latencias = [r["latencia_s"] for r in resultados if r["status"] == "DONE"]
    if latencias:
        print(f"\nCliente · {len(latencias)} jobs DONE · p50 {statistics.median(latencias):.1f} s · "
              f"p95 {percentil(latencias, 0.95):.1f} s · tiempo total {total:.1f} s (en serie serían {sum(latencias):.0f} s)")
    salida = Path(__file__).resolve().parent.parent / "resultados_carga.json"
    salida.write_text(json.dumps({"total_s": round(total, 1), "jobs": resultados}, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(f"Resultados guardados en {salida.name}")


if __name__ == "__main__":
    asyncio.run(main())
