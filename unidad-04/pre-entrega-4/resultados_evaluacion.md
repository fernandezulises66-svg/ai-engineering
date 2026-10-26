# Resultados reales de `evaluate.py`

Ejecución del 26 de octubre de 2026 contra Pinecone Serverless (índice `ola-erp-docs`, namespace `erp-docs-v1`, 20 chunks) con embeddings `text-embedding-3-small`.

```text
========================================================================
REPORTE DE EVALUACIÓN · 5 preguntas · k = 5
========================================================================
Recuperador                Precision@5    Recall@5       MRR
BM25 (léxico)                     0.40        1.00      0.90
Vectorial (Pinecone)              0.40        0.80      0.70
Híbrido (RRF)                     0.48        1.00      0.74
Híbrido + exactos                 0.48        1.00      0.90

Techo de Precision@5 promedio (por cantidad de chunks relevantes): 0.64

Detalle por pregunta (P = precision, R = recall, posición del doc esperado):
             BM25 (léxico)  Vectorial (Pinecone)         Híbrido (RRF)     Híbrido + exactos
q1           P=0.8 R=1  #2         P=0.8 R=1  #1         P=0.8 R=1  #2         P=0.8 R=1  #2   ¿Qué significa el error ERR-AFIP-102 y c
q2           P=0.4 R=1  #1         P=0.6 R=1  #2         P=0.6 R=1  #1         P=0.6 R=1  #1   ¿Cuánto tiempo puede trabajar un usuario
q3           P=0.2 R=1  #1         P=0.2 R=1  #1         P=0.4 R=1  #1         P=0.4 R=1  #1   ¿En qué momento un pedido de la tienda o
q4           P=0.4 R=1  #1         P=0.4 R=1  #1         P=0.4 R=1  #1         P=0.4 R=1  #1   Subí un cambio al backend pero no se ve 
q5           P=0.2 R=1  #1         P=0.0 R=0   —         P=0.2 R=1  #5         P=0.2 R=1  #1   ¿Qué quiere decir el código OLA-RIB-0042

Resultados completos guardados en resultados_evaluacion.json
```

## Corrida anterior (antes de agregar la prioridad a identificadores)

| Recuperador | Precision@5 | Recall@5 | MRR |
|---|---|---|---|
| BM25 (léxico) | 0,40 | 1,00 | 0,90 |
| Vectorial (Pinecone) | 0,40 | 0,80 | 0,70 |
| Híbrido (RRF) | 0,44 | 0,80 | 0,70 |

En esa corrida, el híbrido puro **no** recuperó el documento de q5 (`OLA-RIB-0042`); en la actual lo trae en el puesto 5. La diferencia viene de pequeñas variaciones de los embeddings entre llamadas, y muestra que el híbrido puro queda al límite en esa pregunta.
