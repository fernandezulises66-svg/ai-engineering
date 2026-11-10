"""
Tests sin API keys ni servicios externos:
- Redis → fakeredis (misma API async que redis-py).
- Checkpointer → MemorySaver (en producción, AsyncRedisSaver; mismo contrato de LangGraph).
- LLMs → modelos falsos por rol que deciden según lo que reciben.
La API, la cola, el worker, el grafo, el interrupt del HITL y el manejo de errores son los reales.

    pytest -v
"""

import time
from typing import Any

import fakeredis
import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import MemorySaver

from app.graph import build_graph
from app.hitl import detectar_criticidad
from app.jobs import ERP_ACCIONES
from app.knowledge import LocalKB
from app.main import create_app


class FakeRole(BaseChatModel):
    """Modelo falso. El supervisor delega según la pizarra; los demás devuelven un texto fijo."""

    rol: str
    demora: float = 0.0
    falla: bool = False

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        time.sleep(self.demora)
        if self.falla:
            raise RuntimeError("el proveedor del LLM no responde")
        if self.rol == "supervisor":
            pizarra = messages[-1].content
            siguiente = ("investigador" if "INVESTIGADOR (" not in pizarra
                         else "analista" if "ANALISTA (" not in pizarra else "FINALIZAR")
            texto = f'{{"siguiente": "{siguiente}", "instruccion": "hacé tu parte", "motivo": "test"}}'
        else:
            texto = {"investigador": "- Un presupuesto vale 15 días corridos [03-api-ventas.md]",
                     "analista": "13 días: vigente. Precio: $967.600.",
                     "sintetizador": "Sigue vigente; queda en $967.600."}[self.rol]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(texto))])


def modelos(demora: float = 0.0, falla_en: str | None = None) -> dict[str, FakeRole]:
    return {r: FakeRole(rol=r, demora=demora, falla=(r == falla_en))
            for r in ("supervisor", "investigador", "analista", "sintetizador")}


@pytest.fixture
def redis():
    return fakeredis.aioredis.FakeRedis()


def cliente(redis, **kw) -> TestClient:
    app = create_app(redis_factory=lambda: redis,
                     graph_factory=lambda store: build_graph(MemorySaver(), store.registrar_accion_erp,
                                                             modelos(**kw), LocalKB()))
    return TestClient(app)


def esperar(c: TestClient, job_id: str, hasta: set[str], timeout: float = 10) -> dict[str, Any]:
    fin = time.time() + timeout
    while time.time() < fin:
        job = c.get(f"/tasks/{job_id}").json()
        if job["status"] in hasta:
            return job
        time.sleep(0.05)
    raise AssertionError(f"el job no llegó a {hasta}: {job}")


CONSULTA = "Hoy es 10/11/2026. Un presupuesto del 28/10/2026, ¿sigue vigente?"
CRITICA = "Aplicá un 18 % de descuento al presupuesto P-1042 de $1.180.000 y registralo en el ERP."


def test_post_encola_y_responde_sin_esperar_al_agente(redis):
    with cliente(redis, demora=0.5) as c:          # cada llamada al "LLM" tarda 0,5 s
        inicio = time.perf_counter()
        r = c.post("/tasks", json={"pregunta": CONSULTA})
        assert time.perf_counter() - inicio < 0.3   # responde antes de que el agente haga nada
        assert r.status_code == 202 and r.json()["status"] == "PENDING"
        job = esperar(c, r.json()["job_id"], {"RUNNING"})
        assert job["status"] == "RUNNING"
        assert c.get("/health").json()["status"] == "ok"   # la API sigue respondiendo mientras tanto


def test_consulta_termina_en_done_con_resultado(redis):
    with cliente(redis) as c:
        job_id = c.post("/tasks", json={"pregunta": CONSULTA}).json()["job_id"]
        job = esperar(c, job_id, {"DONE", "FAILED"})
    assert job["status"] == "DONE"
    assert job["resultado"]["respuesta_final"] == "Sigue vigente; queda en $967.600."
    assert job["resultado"]["recorrido"] == ["investigador", "analista", "sintetizador"]
    assert job["resultado"]["es_critica"] is False
    assert [e["status"] for e in job["eventos"]] == ["PENDING", "RUNNING", "DONE"]


def test_tarea_critica_pausa_hasta_la_aprobacion_y_ejecuta_la_accion(redis):
    with cliente(redis) as c:
        job_id = c.post("/tasks", json={"pregunta": CRITICA}).json()["job_id"]
        job = esperar(c, job_id, {"WAITING_APPROVAL", "DONE", "FAILED"})
        assert job["status"] == "WAITING_APPROVAL"
        assert job["aprobacion_pendiente"]["accion_propuesta"]["tipo"] == "aplicar_descuento"
        time.sleep(0.3)
        assert c.get(f"/tasks/{job_id}").json()["status"] == "WAITING_APPROVAL"   # sigue pausado
        assert c.portal.call(redis.llen, ERP_ACCIONES) == 0                         # nada tocó el ERP

        r = c.post(f"/tasks/{job_id}/approve", json={"aprobado": True, "revisor": "gerencia", "comentario": "ok"})
        assert r.status_code == 202
        job = esperar(c, job_id, {"DONE", "FAILED"})
        assert job["status"] == "DONE"
        assert job["resultado"]["aprobacion"]["revisor"] == "gerencia"
        assert "registrada en el ERP" in job["resultado"]["accion_resultado"]
        assert c.portal.call(redis.llen, ERP_ACCIONES) == 1
        assert [e["status"] for e in job["eventos"]] == ["PENDING", "RUNNING", "WAITING_APPROVAL", "PENDING",
                                                         "RUNNING", "DONE"]


def test_rechazo_humano_no_toca_el_erp(redis):
    with cliente(redis) as c:
        job_id = c.post("/tasks", json={"pregunta": CRITICA}).json()["job_id"]
        esperar(c, job_id, {"WAITING_APPROVAL"})
        c.post(f"/tasks/{job_id}/approve", json={"aprobado": False, "revisor": "gerencia"})
        job = esperar(c, job_id, {"DONE", "FAILED"})
        assert "rechazada" in job["resultado"]["accion_resultado"]
        assert c.portal.call(redis.llen, ERP_ACCIONES) == 0


def test_aprobar_un_job_que_no_espera_aprobacion_da_409(redis):
    with cliente(redis) as c:
        job_id = c.post("/tasks", json={"pregunta": CONSULTA}).json()["job_id"]
        esperar(c, job_id, {"DONE"})
        assert c.post(f"/tasks/{job_id}/approve", json={"aprobado": True}).status_code == 409
        assert c.post("/tasks/no-existe/approve", json={"aprobado": True}).status_code == 404
        assert c.get("/tasks/no-existe").status_code == 404


def test_excepcion_en_el_agente_deja_el_job_en_failed(redis):
    with cliente(redis, falla_en="analista") as c:
        job_id = c.post("/tasks", json={"pregunta": CONSULTA}).json()["job_id"]
        job = esperar(c, job_id, {"DONE", "FAILED"})
    assert job["status"] == "FAILED"
    assert "el proveedor del LLM no responde" in job["error"]


def test_cinco_tareas_concurrentes(redis):
    with cliente(redis, demora=0.2) as c:
        inicio = time.perf_counter()
        ids = [c.post("/tasks", json={"pregunta": f"{CONSULTA} (#{i})"}).json()["job_id"] for i in range(5)]
        jobs = [esperar(c, i, {"DONE", "FAILED"}, timeout=20) for i in ids]
        duracion = time.perf_counter() - inicio
    assert all(j["status"] == "DONE" for j in jobs)
    # cada job hace 6 llamadas de 0,2 s (≈1,2 s); en serie serían ≈6 s, en paralelo ≈1,2 s
    assert duracion < 4


def test_validacion_de_entrada(redis):
    with cliente(redis) as c:
        assert c.post("/tasks", json={"pregunta": "hi"}).status_code == 422


@pytest.mark.parametrize("pregunta, critica", [
    ("Aplicá un 18 % de descuento y registralo en el ERP.", True),
    ("Por favor anulá la venta 5042.", True),
    ("emitilo hoy", True),
    ("Un cliente quiere confirmarlo con un 18 % de descuento. ¿Sigue vigente?", False),
    ("¿Qué rol puede aplicar descuentos?", False),
    ("¿Qué significa el error CAJA-422-JUST?", False),
])
def test_deteccion_de_tareas_criticas(pregunta, critica):
    assert detectar_criticidad(pregunta)[0] is critica
