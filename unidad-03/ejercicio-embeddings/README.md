# Ejercicio · Relevancia semántica vs. coincidencia de palabras clave

El entregable es un **PDF**. Este script genera los números del análisis.

`similitud.py` compara la consulta *"Despliegue de microservicios"* contra 7 oraciones (5 relevantes con vocabulario distinto y 2 trampa que comparten palabras clave), con dos métodos:

- **TF-IDF + similitud coseno** (léxico): solo detecta palabras compartidas.
- **Embeddings de OpenAI + similitud coseno** (semántico): capta el significado.

```bash
python similitud.py            # requiere OPENAI_API_KEY en unidad-03/.env
python similitud.py --offline  # solo TF-IDF
```

El cálculo de similitud coseno con scikit-learn:

```python
from sklearn.metrics.pairwise import cosine_similarity

vectores = np.array(embeddings.embed_documents([consulta, *oraciones]))
similitudes = cosine_similarity(vectores[0:1], vectores[1:])[0]
```
