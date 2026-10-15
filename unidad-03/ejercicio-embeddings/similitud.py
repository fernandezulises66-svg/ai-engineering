"""
Unidad 3 · Ejercicio: relevancia semántica vs. coincidencia de palabras clave.

Compara la similitud coseno entre una consulta y 7 oraciones con dos métodos:
  1. TF-IDF (léxico): solo "ve" palabras compartidas.
  2. Embeddings de OpenAI (semántico): capta el significado.

    python similitud.py            # TF-IDF + embeddings (requiere OPENAI_API_KEY)
    python similitud.py --offline  # solo TF-IDF

Guarda los resultados en resultados.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CONSULTA = "Despliegue de microservicios"

ORACIONES = [
    # Mismo concepto, vocabulario distinto
    ("relevante", "Empaquetamos cada servicio en un contenedor Docker y lo publicamos en el clúster de Kubernetes."),
    ("relevante", "La aplicación se divide en componentes independientes que se ponen en producción por separado."),
    ("relevante", "Cada módulo del backend sale a producción con su propio pipeline de CI/CD, sin afectar al resto del sistema."),
    ("relevante", "Subimos la nueva versión de la API de pagos con una estrategia blue-green para no cortar el tráfico."),
    ("relevante", "Orquestamos decenas de pequeñas APIs autónomas que escalan horizontalmente según la demanda."),
    # Trampas: comparten palabras clave, pero no el significado
    ("trampa", "El servicio de micro-limpieza de alfombras es excelente y muy puntual."),
    ("trampa", "El despliegue de la bandera en el acto escolar fue muy emotivo."),
]


STOPWORDS = ["de", "la", "el", "en", "y", "a", "los", "las", "un", "una", "con", "por", "para",
             "que", "se", "lo", "su", "sin", "al", "del", "es", "fue", "muy", "cada"]


def similitud_tfidf(consulta: str, textos: list[str]) -> np.ndarray:
    vectorizer = TfidfVectorizer(lowercase=True, strip_accents="unicode", stop_words=STOPWORDS)
    matriz = vectorizer.fit_transform([consulta, *textos])
    return cosine_similarity(matriz[0:1], matriz[1:])[0]


def similitud_embeddings(consulta: str, textos: list[str], modelo: str = "text-embedding-3-small") -> np.ndarray:
    from dotenv import find_dotenv, load_dotenv
    from langchain_openai import OpenAIEmbeddings

    load_dotenv(find_dotenv())
    emb = OpenAIEmbeddings(model=modelo)
    vectores = np.array(emb.embed_documents([consulta, *textos]))  # mismo modelo para todo
    return cosine_similarity(vectores[0:1], vectores[1:])[0]


def main() -> None:
    textos = [t for _, t in ORACIONES]
    resultados = {"consulta": CONSULTA, "tfidf": similitud_tfidf(CONSULTA, textos).round(3).tolist()}
    if "--offline" not in sys.argv:
        resultados["embeddings"] = similitud_embeddings(CONSULTA, textos).round(3).tolist()
        resultados["modelo_embeddings"] = "text-embedding-3-small"

    print(f'Consulta: "{CONSULTA}"\n')
    print(f"{'#':<3}{'tipo':<11}{'TF-IDF':>8}{'Embed.':>8}  oración")
    for i, (tipo, texto) in enumerate(ORACIONES):
        emb = f"{resultados['embeddings'][i]:.3f}" if "embeddings" in resultados else "  —"
        print(f"{i + 1:<3}{tipo:<11}{resultados['tfidf'][i]:>8.3f}{emb:>8}  {texto[:70]}")

    resultados["oraciones"] = [{"tipo": t, "texto": x} for t, x in ORACIONES]
    salida = Path(__file__).with_name("resultados.json")
    salida.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResultados guardados en {salida.name}")


if __name__ == "__main__":
    main()
