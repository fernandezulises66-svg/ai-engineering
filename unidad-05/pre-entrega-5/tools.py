"""
Herramientas del agente (contrato con el LLM).

El modelo elige qué herramienta usar leyendo SOLO el nombre, el docstring y los
tipos de los parámetros. Por eso los docstrings dicen qué hace cada una, cuándo
usarla, qué devuelve y qué hacer si falla.

Las herramientas devuelven JSON. Los errores NO se lanzan como excepción: se
devuelven como {"error": ..., "sugerencia": ...} para que el agente pueda
corregirse y volver a intentar (ciclo de retorno).
"""

import json
import unicodedata
from typing import Any, Literal

from langchain_core.tools import tool

import db

Estado = Literal["pendiente_pago", "facturado", "en_fabricacion", "entregado", "cancelado"]


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False)


def _normalizar(texto: str) -> list[str]:
    texto = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in texto if not unicodedata.combining(c)).split()


@tool
async def buscar_cliente(nombre: str) -> str:
    """Busca clientes del ERP de OLA Muebles por nombre o apellido y devuelve su cliente_id.

    Usala SIEMPRE antes de consultar pedidos cuando el usuario nombra a un cliente
    pero no da su número de cliente (cliente_id). La búsqueda no distingue acentos
    ni mayúsculas, y exige que cada palabra buscada coincida con una palabra completa
    del nombre ("Gómez" encuentra "Martina Gómez"; "Martín" NO encuentra "Martina").

    Devuelve JSON con la lista `clientes` (cliente_id, nombre, localidad).
    - Si hay más de un resultado, NO elijas uno al azar: preguntale al usuario cuál es
      (podés usar la localidad para diferenciarlos).
    - Si no hay resultados, probá de nuevo con menos palabras (por ejemplo, solo el
      apellido) antes de responder que el cliente no existe.
    """
    buscadas = _normalizar(nombre)
    if not buscadas:
        return _json({"error": "El nombre está vacío.", "sugerencia": "Pedile al usuario el nombre del cliente."})
    clientes = await db.query("SELECT id AS cliente_id, nombre, localidad FROM clientes ORDER BY id")
    encontrados = [c for c in clientes if all(p in _normalizar(c["nombre"]) for p in buscadas)]
    if not encontrados:
        return _json({
            "clientes": [],
            "error": f"No hay clientes que coincidan con '{nombre}'.",
            "sugerencia": "Reintentá buscando solo por el apellido o revisá la ortografía.",
        })
    return _json({"clientes": encontrados, "cantidad": len(encontrados)})


@tool
async def buscar_pedidos(cliente_id: int, estado: Estado | None = None) -> str:
    """Devuelve los pedidos de un cliente: cantidad, total en pesos y la lista ordenada por fecha.

    Requiere el cliente_id numérico (obtenelo antes con `buscar_cliente` si solo tenés
    el nombre). Opcionalmente filtra por `estado`: pendiente_pago, facturado,
    en_fabricacion, entregado o cancelado.

    Devuelve JSON con `cantidad`, `total` (suma de los pedidos que no están cancelados),
    `pedidos` (pedido_id, fecha, estado, total; del más viejo al más nuevo) y
    `ultimo_pedido_id`. Si el cliente no existe, devuelve un `error`.
    """
    clientes = await db.query("SELECT id, nombre FROM clientes WHERE id = ?", (cliente_id,))
    if not clientes:
        return _json({"error": f"No existe el cliente_id {cliente_id}.",
                      "sugerencia": "Usá buscar_cliente para obtener el cliente_id correcto."})
    sql = "SELECT id AS pedido_id, fecha, estado, total FROM pedidos WHERE cliente_id = ?"
    params: tuple[Any, ...] = (cliente_id,)
    if estado:
        sql, params = sql + " AND estado = ?", (cliente_id, estado)
    pedidos = await db.query(sql + " ORDER BY fecha", params)
    total = sum(p["total"] for p in pedidos if p["estado"] != "cancelado")
    return _json({
        "cliente_id": cliente_id,
        "cliente": clientes[0]["nombre"],
        "cantidad": len(pedidos),
        "total": total,
        "pedidos": pedidos,
        "ultimo_pedido_id": pedidos[-1]["pedido_id"] if pedidos else None,
    })


@tool
async def detalle_pedido(pedido_id: int) -> str:
    """Devuelve el detalle de un pedido: fecha, estado, cliente y los ítems (SKU, descripción, cantidad, precio).

    Usala cuando el usuario pregunta qué incluye un pedido puntual o por "el último
    pedido" (en ese caso, tomá `ultimo_pedido_id` de una consulta previa a
    `buscar_pedidos`). Requiere el pedido_id numérico; si no existe, devuelve un `error`.
    """
    pedidos = await db.query(
        "SELECT p.id AS pedido_id, p.fecha, p.estado, p.total, c.nombre AS cliente "
        "FROM pedidos p JOIN clientes c ON c.id = p.cliente_id WHERE p.id = ?", (pedido_id,))
    if not pedidos:
        return _json({"error": f"No existe el pedido {pedido_id}.",
                      "sugerencia": "Consultá buscar_pedidos para ver los pedido_id válidos del cliente."})
    items = await db.query(
        "SELECT sku, descripcion, cantidad, precio_unitario FROM items WHERE pedido_id = ?", (pedido_id,))
    return _json({**pedidos[0], "items": items})


TOOLS = [buscar_cliente, buscar_pedidos, detalle_pedido]
