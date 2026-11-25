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

Si el entorno existe, usar `conda env update -f environment.yml --prune`. La primera ejecución descarga 20 Newsgroups desde la fuente oficial de scikit-learn; no se versiona el corpus completo. Para ejecutar únicamente una etapa: `python -m proyecto experiment001` (también 002, 003, 004 o `entrega`). Ejecutar todas las etapas antes de generar la entrega. Para clasificar textos nuevos después del experimento003:

```powershell
python -m proyecto --texto "The spacecraft entered orbit around Mars."
```

## Diseño experimental

Seis temas distintos: gráficos por computadora, automóviles, hockey, medicina, espacio y política de Oriente Medio. Se seleccionan 400 documentos de entrenamiento y 150 de prueba por categoría (3,300 en total), respetando las particiones oficiales, con semilla 42. Se eliminan cabeceras, firmas, citas, documentos sin vocabulario útil y duplicados exactos tras limpieza. La prueba se mantiene fuera del ajuste supervisado y de los estudios de generalización. El agrupamiento es un análisis independiente sobre el corpus completo: sus ajustes no reciben etiquetas ni se transfieren al clasificador.

La selección supervisada se realiza con F1 macro y los mismos cinco pliegues estratificados. GridSearchCV usa n_jobs=-1; los estimadores internos y BLAS se limitan durante la búsqueda para evitar sobreparalelización. Se mantiene scikit-learn en CPU: estos estimadores no ofrecen GPU nativa en la configuración usada. No se introduce una implementación CUDA alternativa por el tamaño del corpus y la comparabilidad.

| Experimento | Objetivo | Módulo |
| --- | --- | --- |
| experiment001 | EDA, BoW, TF-IDF y vocabulario | preparacion.py |
| experiment002 | K-means, aglomerativo, DBSCAN, mezcla gaussiana EM y SVD | agrupamiento.py |
| experiment003 | Siete clasificadores, CV, prueba final, bootstrap y Holm | clasificacion.py |
| experiment004 | Representación, dimensión y sobreajuste con CV | generalizacion.py |

La ejecución completa realiza 001, 002, 004 y 003 (el estudio de generalización se termina antes de evaluar prueba). Los resultados se guardan en resultados/experimentNNN. Después se generan notebook_final.ipynb (con salidas) y reporte.pdf, en cinco capítulos, a partir de los resultados reales. No se inventan métricas. El reporte se genera automáticamente; debe revisarse visual y científicamente antes de entregar.

## Runner

El workflow de GitHub Actions está configurado para un runner propio Windows x64 con Conda/Miniforge. Se activa al publicar cambios de código en la rama de experimentos y permite ejecución manual desde la interfaz de Actions cuando el workflow esté disponible. Conserva resultados, notebook y PDF como artefactos, e intenta versionarlos en la misma rama si el token tiene permiso. No modifica main.

Fuente: https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset
