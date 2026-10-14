"""Contrato de salida del pipeline: el objeto validado que devuelve la cadena."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class NivelCriticidad(str, Enum):
    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class ExtraccionTecnica(BaseModel):
    """Entidades técnicas extraídas de un texto (log de error, descripción de arquitectura, etc.)."""

    model_config = ConfigDict(extra="forbid")

    tecnologias: list[str] = Field(
        min_length=1,
        max_length=20,
        description=(
            "Tecnologías, lenguajes, frameworks, bases de datos o servicios mencionados "
            "explícitamente en el texto, con su nombre propio (ej. 'FastAPI', 'Redis')."
        ),
    )
    nivel_de_criticidad: NivelCriticidad = Field(
        description=(
            "alta: caída del servicio, pérdida de datos o riesgo de seguridad; "
            "media: degradación de rendimiento o errores parciales; "
            "baja: información, advertencias o descripciones sin impacto inmediato."
        )
    )
    resumen_tecnico: str = Field(
        min_length=20,
        max_length=400,
        description="Resumen técnico en una o dos oraciones, en español.",
    )

    @field_validator("tecnologias")
    @classmethod
    def limpiar_tecnologias(cls, valores: list[str]) -> list[str]:
        """Quita espacios, vacíos y duplicados (sin distinguir mayúsculas), conservando el orden."""
        vistos: set[str] = set()
        limpias: list[str] = []
        for valor in valores:
            nombre = valor.strip()
            if nombre and nombre.lower() not in vistos:
                vistos.add(nombre.lower())
                limpias.append(nombre)
        if not limpias:
            raise ValueError("la lista de tecnologías no puede quedar vacía")
        return limpias
