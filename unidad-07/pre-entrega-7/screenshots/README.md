# Capturas del dashboard de Arize Phoenix

Corrida de 5 peticiones concurrentes (`python scripts/carga.py --aprobar`). La lectura de cada captura está en el [README principal](../README.md#resultados-y-capturas).

| Archivo | Qué muestra |
|---|---|
| `01-trazas.png` | Las 6 trazas del proyecto `ola-erp-orquestador` (5 `job.start` + 1 `job.resume`) |
| `02a-traza-supervisor.png` … `02d-traza-sintetizador.png` | Árbol de una traza nodo por nodo, con tokens y costo por llamada al LLM |
| `03a-hitl-pausa.png` | La tarea crítica pausada en `aprobacion_humana` |
| `03b-hitl-sesion.png` | Sesión del job: pausa (`WAITING_APPROVAL`) y reanudación |
| `04-costo.png` / `04b-costo-por-modelo.png` | Costo de la corrida (entrada y salida) y costo por modelo |
| `05-latencia-p95.png` | Latencia p95 de las 5 ejecuciones: 60,01 s |
