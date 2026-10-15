# OLA Muebles — Manual interno del ERP: presupuestos, ventas y caja

> Documento ficticio, creado con fines educativos para el curso de AI Engineering. Describe procedimientos de ejemplo y no corresponde a un sistema en producción.

## Roles y permisos

El ERP tiene roles con permisos distintos. Los más usados en el área comercial son:

- **Vendedor:** crea presupuestos, los convierte en ventas y registra cobros. No puede anular comprobantes ni modificar precios del catálogo.
- **Encargado de local:** además de lo anterior, puede aplicar descuentos de hasta el 15 %, anular ventas del día y hacer el cierre de caja.
- **Administrador:** acceso total, incluida la anulación de comprobantes de días anteriores, los descuentos mayores al 15 % y la configuración de usuarios.

La sesión del ERP vence a las 8 horas. Si se vence en medio de una carga, los datos no guardados se pierden, por eso conviene guardar el presupuesto como borrador antes de consultar precios en otra pestaña.

## Crear un presupuesto

1. Ir a **Presupuestos → Nuevo presupuesto**.
2. Buscar al cliente por nombre, CUIT o DNI. Si no existe, crearlo desde el botón "Nuevo cliente"; los campos obligatorios son nombre, condición frente al IVA y un medio de contacto.
3. Agregar los productos desde el catálogo. Para muebles de la Línea Atelier, usar el producto genérico "Mueble a medida" y detallar medidas, madera y terminación en el campo de observaciones.
4. Indicar la zona de entrega para que el sistema sume el costo de envío correspondiente.
5. Guardar. El presupuesto tiene una validez de 15 días corridos, y después de ese plazo los precios deben actualizarse.
6. Exportar el presupuesto en PDF desde el botón "Descargar PDF" y enviarlo al cliente por correo.

## Convertir un presupuesto en venta

Cuando el cliente confirma, abrir el presupuesto y elegir **Convertir en venta**. El sistema copia los productos y pide completar los datos fiscales:

- **Tipo de comprobante:** factura A si el cliente es responsable inscripto en IVA; factura B si es consumidor final, monotributista o exento.
- **Condición de venta:** contado, cuenta corriente o seña.
- **Punto de venta:** el 0003 corresponde al showroom de City Bell y el 0005 a la tienda online.

Para los muebles a medida, la venta se registra con condición "seña" y se cobra el 50 % al confirmar. El saldo se cobra antes de la entrega; ningún mueble a medida sale del taller con saldo pendiente.

Las ventas de la tienda online se crean solas, con el usuario "Tienda online", únicamente cuando el pago está acreditado. Los pedidos pendientes de pago no generan venta ni reservan stock.

## Registrar un cobro en caja

El módulo de caja tiene dos cuentas: **Efectivo** y **Banco**. Las transferencias, los depósitos y los cobros con tarjeta se registran siempre en Banco; los pagos en billetes, en Efectivo.

Al registrar un cobro de una venta, el sistema pregunta si se quiere generar el movimiento de caja asociado. Siempre hay que responder que sí, para que el saldo de caja coincida con las ventas cobradas. Si un cobro se registró en la cuenta equivocada, el vendedor no puede corregirlo: debe pedirle al encargado de local que haga un movimiento de ajuste entre cuentas, con una observación que explique el motivo.

## Cierre de caja diario

El cierre de caja lo hace el encargado de local al final de cada jornada, desde **Caja → Cierre del día**:

1. Contar el efectivo físico y cargarlo en el campo "Efectivo contado".
2. El sistema compara ese valor con el saldo teórico de la cuenta Efectivo y muestra la diferencia.
3. Si la diferencia supera los $5.000, es obligatorio escribir una justificación antes de cerrar.
4. Confirmar el cierre. Una vez cerrado, ningún usuario puede registrar movimientos con fecha de ese día; las correcciones posteriores se hacen con un movimiento de ajuste en la fecha actual.

El informe de cierre se descarga en PDF y se archiva en la carpeta compartida de administración.

## Anulaciones y notas de crédito

Una venta mal cargada en el día puede anularla el encargado de local. Si la venta es de un día anterior y ya se emitió la factura electrónica, no se anula: el administrador debe emitir una nota de crédito del mismo tipo (A o B) que la factura original y, si corresponde, una nueva factura con los datos correctos.
