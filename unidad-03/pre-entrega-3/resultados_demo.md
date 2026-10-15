# Resultados reales de `demo.py`

Ejecución del 15 de octubre de 2026 con `claude-haiku-5-5` (generación) y `text-embedding-3-small` (embeddings). Las 4 pruebas dieron el resultado esperado.

```text

================================================================================
[en_documentos] Se me rompió una bisagra del rack a los 4 meses. ¿Está en garantía y cómo hago el reclamo?
{
  "pregunta": "Se me rompió una bisagra del rack a los 4 meses. ¿Está en garantía y cómo hago el reclamo?",
  "respuesta": "Sí, en principio está en garantía: los herrajes, como las bisagras, tienen 6 meses de cobertura contados desde la fecha de entrega del remito, y tu caso ocurrió a los 4 meses. Para hacer el reclamo, tenés que enviar un correo a info@olamuebles.com con asunto Garantía – número de pedido, incluyendo el número de pedido o comprobante, una descripción del problema y al menos tres fotos (del mueble completo, del detalle del defecto y de la etiqueta del producto). Posventa responde en hasta 72 horas hábiles; si se resuelve con un repuesto, este se envía sin cargo, junto con un video instructivo. La garantía cubre defectos de fabricación, así que el equipo evaluará si el problema corresponde a eso o a alguna de las causas excluidas.",
  "encontrada_en_contexto": true,
  "referencias": [
    {
      "fuente": "02-garantia-y-devoluciones.md",
      "titulo": "OLA Muebles — Garantía, cambios y devoluciones",
      "chunk": 1,
      "score": 0.47,
      "extracto": "# OLA Muebles — Garantía, cambios y devoluciones > Documento ficticio, creado con fines educativos para el curso de AI Engineering. Los datos no representan condiciones comerciales reales. ## Cobertura de la garantía Todos los muebles de OL…"
    },
    {
      "fuente": "02-garantia-y-devoluciones.md",
      "titulo": "OLA Muebles — Garantía, cambios y devoluciones",
      "chunk": 2,
      "score": 0.46,
      "extracto": "## Qué no cubre la garantía La garantía no cubre los daños causados por: - Exposición prolongada a la humedad, al agua o a fuentes de calor (estufas, radiadores, hornallas). - Exposición directa al sol, que puede provocar cambios de tono en…"
    }
  ]
}
→ OK: encontrada_en_contexto=True (esperado True)

================================================================================
[en_documentos] ¿Puedo devolver un placar a medida si me arrepiento?
{
  "pregunta": "¿Puedo devolver un placar a medida si me arrepiento?",
  "respuesta": "No, si el placard es fabricado a medida (Línea Atelier) no podés arrepentirte, porque el derecho de arrepentimiento no aplica a esos muebles. Además, una vez que aprobaste el plano y empezó la fabricación, la seña no es reembolsable. La documentación no aclara si un placard a medida que no sea de la Línea Atelier tendría el mismo tratamiento.",
  "encontrada_en_contexto": true,
  "referencias": [
    {
      "fuente": "02-garantia-y-devoluciones.md",
      "titulo": "OLA Muebles — Garantía, cambios y devoluciones",
      "chunk": 4,
      "score": 0.554,
      "extracto": "## Derecho de arrepentimiento Para compras realizadas en la tienda online, por correo electrónico o por WhatsApp (es decir, a distancia), el cliente tiene **10 días corridos** desde la entrega para arrepentirse de la compra, sin necesidad d…"
    }
  ]
}
→ OK: encontrada_en_contexto=True (esperado True)

================================================================================
[trampa] ¿En cuántas cuotas sin interés puedo pagar con tarjeta de crédito?
{
  "pregunta": "¿En cuántas cuotas sin interés puedo pagar con tarjeta de crédito?",
  "respuesta": "No lo sé: esa información no está en la documentación disponible.",
  "encontrada_en_contexto": false,
  "referencias": []
}
→ OK: encontrada_en_contexto=False (esperado False)

================================================================================
[trampa] ¿Cuántos empleados tiene OLA Muebles y quién es el dueño?
{
  "pregunta": "¿Cuántos empleados tiene OLA Muebles y quién es el dueño?",
  "respuesta": "No lo sé: esa información no está en la documentación disponible.",
  "encontrada_en_contexto": false,
  "referencias": []
}
→ OK: encontrada_en_contexto=False (esperado False)
```
