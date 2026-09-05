# Balance de deforestación y recuperación — bosque manejado de Quebec

Análisis del balance neto entre pérdida y recuperación de cobertura forestal en el área de bosque público bajo aprovechamiento en Quebec, usando series temporales de Sentinel-2 en Google Earth Engine.

## Pregunta de investigación

¿Cuál es el balance neto anual entre superficie perdida (corte, fuego, plagas) y superficie en recuperación dentro del bosque manejado, y cómo varía espacialmente?

## Datos

- **Imágenes:** Sentinel-2 SR Harmonized (2017–2025), composites anuales de temporada de crecimiento (junio–septiembre).
- **Área de estudio:** región administrativa Abitibi-Témiscamingue (08), completa. Límites de la capa "Découpages administratifs" de Données Québec (o unión de las 5 divisions de recensement equivalentes de Statistique Canada 2021).
- **Unidad espacial de agregación:** [pendiente — grilla de hexágonos o subdivisión de UAF].

## Metodología

1. Extracción de composites anuales con máscara de nubes (SCL) — `src/gee_utils.py`.
2. Cálculo de NDVI y NBR por composite.
3. Detección de cambio año contra año (dNBR) y clasificación en pérdida / recuperación / estable — `src/change_detection.py`.
4. Agregación espacial: superficie de pérdida y recuperación (ha) por unidad, balance neto = recuperación − pérdida.
5. Visualización: mapas anuales y gráfico de balance neto acumulado.

## Estructura

```
src/                  funciones reusables (GEE, detección de cambio)
notebooks/            flujo de análisis y resultados
figures/              mapas y gráficos finales
data/                 capas públicas o instrucciones de descarga
```

## Cómo correrlo

1. `pip install -r requirements.txt`
2. `earthengine authenticate`
3. Descargar la capa de "Découpages administratifs" (Données Québec) y colocarla en `data/`.
4. Reemplazar `PROJECT` en el notebook por tu proyecto de GEE.
5. Definir la capa de unidades espaciales para la agregación.
6. Correr `notebooks/01_extraccion_y_balance.ipynb` en orden.

## Nota metodológica

Los umbrales de dNBR para clasificar pérdida/recuperación son de partida y deben calibrarse contra eventos de perturbación conocidos (ej. perímetros de incendio mapeados por SOPFEU) antes de interpretar resultados como definitivos.

## Próximo paso

Este repo es la primera de dos publicaciones. La segunda parte usa el balance neto por unidad como insumo de un modelo de optimización (MILP) para priorizar restauración bajo restricción de presupuesto.

