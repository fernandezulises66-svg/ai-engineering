"""
Base de datos SQLite de demostración del ERP de OLA Muebles (datos ficticios).

Se crea sola la primera vez. Las consultas son asíncronas (aiosqlite), como lo
serían contra una base real, para no bloquear el event loop del agente.
"""

from pathlib import Path
from typing import Any

import aiosqlite

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS clientes (
    id INTEGER PRIMARY KEY, nombre TEXT NOT NULL, localidad TEXT NOT NULL, email TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pedidos (
    id INTEGER PRIMARY KEY, cliente_id INTEGER NOT NULL REFERENCES clientes(id),
    fecha TEXT NOT NULL, estado TEXT NOT NULL, total REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY, pedido_id INTEGER NOT NULL REFERENCES pedidos(id),
    sku TEXT NOT NULL, descripcion TEXT NOT NULL, cantidad INTEGER NOT NULL, precio_unitario REAL NOT NULL
);
"""

CLIENTES = [
    (102, "Martina Gómez", "City Bell", "martina.gomez@example.com"),
    (103, "Joaquín Gómez", "Gonnet", "joaquin.gomez@example.com"),
    (104, "Lucía Fernández", "La Plata", "lucia.f@example.com"),
    (105, "Estudio Ribera SRL", "CABA", "compras@estudioribera.example.com"),
]

PEDIDOS = [  # (id, cliente_id, fecha, estado, total)
    (5001, 102, "2026-06-14", "entregado", 4200.0),
    (5017, 102, "2026-08-02", "entregado", 3800.0),
    (5042, 102, "2026-10-21", "en_fabricacion", 6500.0),
    (5008, 103, "2026-07-01", "entregado", 980.0),
    (5031, 104, "2026-09-12", "cancelado", 2100.0),
    (5039, 105, "2026-10-03", "facturado", 18750.0),
    (5044, 105, "2026-10-30", "pendiente_pago", 9200.0),
]

ITEMS = [  # (pedido_id, sku, descripcion, cantidad, precio_unitario)
    (5001, "OLA-RIB-0042", "Mesa Ribera 1,80 m petiribí", 1, 4200.0),
    (5017, "OLA-BEL-0007", "Rack Bell 1,60 m paraíso", 1, 3800.0),
    (5042, "OLA-ATL-0115", "Biblioteca a medida roble americano", 1, 5900.0),
    (5042, "OLA-OUT-0003", "Kit de mantenimiento (aceite + paño)", 2, 300.0),
    (5008, "OLA-OUT-0003", "Kit de mantenimiento (aceite + paño)", 1, 980.0),
    (5031, "OLA-BEL-0011", "Mesa ratona Bell elevable", 1, 2100.0),
    (5039, "OLA-RIB-0050", "Mesa Ribera 2,20 m petiribí", 3, 6250.0),
    (5044, "OLA-RIB-0061", "Silla Ribera tapizada en lino", 8, 1150.0),
]


async def init_db(path: Path = config.DB_PATH) -> None:
    """Crea y carga la base si no existe (idempotente)."""
    async with aiosqlite.connect(path) as db:
        await db.executescript(SCHEMA)
        if (await (await db.execute("SELECT COUNT(*) FROM clientes")).fetchone())[0] == 0:
            await db.executemany("INSERT INTO clientes VALUES (?, ?, ?, ?)", CLIENTES)
            await db.executemany("INSERT INTO pedidos VALUES (?, ?, ?, ?, ?)", PEDIDOS)
            await db.executemany(
                "INSERT INTO items (pedido_id, sku, descripcion, cantidad, precio_unitario) VALUES (?, ?, ?, ?, ?)",
                ITEMS)
            await db.commit()


async def query(sql: str, params: tuple[Any, ...] = (), path: Path | None = None) -> list[dict[str, Any]]:
    async with aiosqlite.connect(path or config.DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await (await db.execute(sql, params)).fetchall()
        return [dict(r) for r in rows]
