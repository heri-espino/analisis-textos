# Análisis y clasificación de textos

Proyecto final en español. Descubrimiento de temas y clasificación de documentos de 20 Newsgroups con scikit-learn.

## Reproducción (Windows, Conda)

```powershell
conda env create -f environment.yml
conda activate analisis-textos
python -m pip install --no-deps -e .
python -m pytest -q
python -m proyecto
```

Si el entorno existe, usar `conda env update -f environment.yml --prune`. La primera ejecución descarga 20 Newsgroups desde la fuente oficial de scikit-learn; no se versiona el corpus completo. Para ejecutar únicamente una etapa: `python -m proyecto experiment001` (también 002, 003, 004, 005, 006, 007 o `entrega`). Ejecutar todas las etapas antes de generar la entrega. Para clasificar textos nuevos después del experimento003:

```powershell
python -m proyecto --texto "The spacecraft entered orbit around Mars."
```

## Diseño experimental

Seis temas distintos: gráficos por computadora, automóviles, hockey, medicina, espacio y política de Oriente Medio. Se seleccionan 400 documentos de entrenamiento y 150 de prueba por categoría (3,300 en total), respetando las particiones oficiales, con semilla 42. Se eliminan cabeceras, firmas, citas, documentos sin vocabulario útil y duplicados exactos tras limpieza. La prueba se mantiene fuera del ajuste supervisado y de los estudios de generalización. El agrupamiento es un análisis independiente sobre el corpus completo: sus ajustes no reciben etiquetas ni se transfieren al clasificador. experiment006 incorpora grupos inductivos reajustados dentro de cada Pipeline; nunca reutiliza las asignaciones globales.

La selección supervisada se realiza con F1 macro y los mismos cinco pliegues estratificados. GridSearchCV usa n_jobs=-1; los estimadores internos y BLAS se limitan durante la búsqueda para evitar sobreparalelización. Se mantiene scikit-learn en CPU: estos estimadores no ofrecen GPU nativa en la configuración usada. XGBoost se utiliza como metaestimador con tree_method=hist en CPU: la matriz de metacaracterísticas es pequeña y no justifica transferencias a GPU. Se registran tiempos de pared de búsqueda, reajuste, predicción y agrupamiento; las comparaciones dependen del hardware. Los ensambles usan bases preespecificadas y CV externa común, con CV interna para stacking.

| Experimento | Objetivo | Módulo |
| --- | --- | --- |
| experiment001 | EDA, BoW, TF-IDF y vocabulario | preparacion.py |
| experiment002 | K-means, aglomerativo, DBSCAN, mezcla gaussiana EM y SVD | agrupamiento.py |
| experiment003 | Siete clasificadores y tres ensambles, CV, prueba final, bootstrap y Holm | clasificacion.py |
| experiment004 | Representación, dimensión y sobreajuste con CV | generalizacion.py |
| experiment005 | Bagging logístico, votación de siete modelos y stacking XGBoost, integrados en 003 | ensambles.py |
| experiment006 | Ablación: mismos diez modelos con 20 indicadores temáticos K-means/GMM | hibrido.py, temas.py |
| experiment007 | Seis proyecciones 2D, dos dashboards, preservación local y tiempos | proyecciones.py |

La ejecución completa realiza 001, 002, 007, 004, 003 (incluye 005) y 006 (el estudio de generalización se termina antes de evaluar prueba). Los resultados se guardan en resultados/experimentNNN. Después se generan notebook_final.ipynb (con salidas) y reporte.pdf, en cinco capítulos, a partir de los resultados reales. No se inventan métricas. El reporte se genera automáticamente; debe revisarse visual y científicamente antes de entregar.

## Runner

El workflow de GitHub Actions está configurado para un runner propio Windows x64 con Conda/Miniforge. Se activa al publicar cambios de código en la rama de experimentos y permite ejecución manual desde la interfaz de Actions cuando el workflow esté disponible. Conserva resultados, notebook y PDF como artefactos, e intenta versionarlos en la misma rama si el token tiene permiso. No modifica main.

Fuente: https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset

## Pregunta integradora

¿La estructura temática no supervisada mejora la clasificación respecto a la representación léxica? experiment006 compara los mismos diez candidatos con y sin 20 columnas one-hot (10 K-means + 10 GMM, SVD-50 normalizado). Se conservan particiones, rejillas y pliegues; todos los ajustes están dentro del pipeline. El corpus de prueba se había evaluado antes de proponer esta extensión: sus resultados son exploratorios, no una nueva prueba independiente. Para ejecutar solo la extensión con resultados previos del 003: `python -m proyecto experiment006`, luego `python -m proyecto entrega`. Para predecir con el mejor híbrido: `python -m proyecto --hibrido --texto "spacecraft orbit"`.

## Visualización de la geometría textual

Para actualizar únicamente los dos dashboards del capítulo 2 con resultados anteriores de experiment002: `python -m proyecto experiment007` y `python -m proyecto entrega`. PCA centrado, ICA, t-SNE, UMAP, Isomap y Spectral Embedding comparten TF-IDF → SVD-100. Se guardan las coordenadas de todos los documentos, tiempos, trustworthiness(15), parámetros y advertencias. Los colores corresponden a categorías reales o grupos seleccionados, únicamente después del ajuste. Las proyecciones son exploratorias y nunca se transfieren al clasificador.

## Diagnósticos complementarios
`python -m proyecto experiment008` reconstruye las curvas de inercia, los dendrogramas de documentos y los pesos de términos, y amplía SVD hasta 1000 dimensiones. Requiere las asignaciones originales del agrupamiento y conserva las métricas históricas. Consultar experimentos/experiment008/README.md.

`python -m proyecto experiment009`: medoides coseno de los diez grupos y etiquetas DBSCAN originales, verificadas contra las métricas históricas.

## Notebook autónoma
`notebook_autonoma.ipynb` contiene el código fuente de todos los módulos necesarios. No requiere clonar el repositorio ni descargar scripts. Ejecutar en el entorno Conda anterior o activar INSTALAR. La primera ejecución descarga 20 Newsgroups. El flujo dedicado valida la sintaxis; para ejecutarla completa, usar workflow_dispatch con ejecutar_completa=true (requiere nbconvert).
