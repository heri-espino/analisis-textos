# Entrega y continuidad

La implementación inicial define experiment001–005. El runner ejecuta los experimentos y genera la entrega reproducible. Antes de declarar concluido el proyecto, comprobar el estado de Actions, revisar las tablas y figuras, confirmar las 3,300 observaciones, las siete familias supervisadas y tres ensambles, los cuatro algoritmos de agrupamiento y tres contrastes de Holm; inspeccionar el PDF y ampliar la interpretación sustantiva si procede. Los resultados definitivos son los archivos versionados o artefactos del runner, nunca una predicción de desempeño.

Si falla el runner, revisar etiquetas Windows/X64, disponibilidad de Conda, acceso a Figshare y permisos del token. Si la descarga falla, reintentar sin cambiar los conjuntos. Las siguientes investigaciones deben usar experiment006, etc. No reajustar basándose en la prueba ya observada.

La primera ejecución 45568371842 falló antes de las pruebas porque no existe pwsh en el runner. Se cambia a Windows PowerShell y se invoca conda.exe directamente. experiment005 está integrado en 003 para congelar el ranking de diez modelos antes de prueba. Revisar los nuevos tiempos y gráficos, sin inferir resultados hasta ejecutarlos.

La segunda ejecución 45569130565 no generó jobs: corregir la sintaxis YAML de los comandos PowerShell con bloques literales. Las métricas de clasificación se grafican separadas de los tiempos.

La tercera ejecución 45569178773 falló por instalación existente en miniconda3. Se reutiliza la instalación local, se detecta conda.exe y se ejecuta conda env update -n analisis-textos -f environment.yml --prune.

experiment006: conservar la primera ejecución exitosa en c1324670b7bbbf15d9e97c7b1b181c9d7ecb227f. Repetir diez candidatos con 20 indicadores temáticos; comparar por algoritmo y construir ranking global. La prueba ya observada es exploratoria. Validar los resultados nuevos y el PDF antes de entregar. Siguiente investigación: experiment007.

experiment006 finalizó correctamente en 9f580fbd1cd76aa0249fee495530499e6731795d. experiment007 agrega seis proyecciones sobre SVD-100 y dos dashboards al capítulo 2. La ejecución de esta extensión reutiliza los resultados previos y no reentrena clasificadores. Siguiente investigación: experiment008. Verificar 19,800 filas (3,300 por método), seis métricas, dos PNG y regeneración de notebook y PDF.

experiment008: repetir únicamente diagnósticos que faltaban en el informe editorial. Verificar los resultados antes de incorporar cifras; no sustituir la prueba histórica. La rama no se fusiona sin autorización.

experiment009: eliminar aproximaciones en 2D. Recalcular medoides y etiquetas DBSCAN en SVD-50, comprobar ARI=1 para K-means y métricas históricas de DBSCAN; usar huella para colorear las seis proyecciones. Conservar resultados supervisados.

experiment010: corte jerárquico descriptivo en diez grupos, categoría dominante y pureza de Gini normalizada por nodo. Se reproduce sin cambiar distancias ni reentrenar; consultar experimentos/experiment010/README.md. El informe editorial se entrega aparte.

Notebook autónoma: código fuente embebido, módulos locales generados por celdas %%writefile, cinco capítulos y gráficos derivados. No sustituye las evaluaciones históricas. Flujo dedicado para verificar sintaxis y ejecución opcional.

Reorganización de la notebook autónoma: fragmentos fuente pequeños, ensamblado reproducible y celdas separadas para ajuste, tablas y figuras. No cambia el protocolo de los experimentos. El flujo de verificación comprueba el ensamblado y carga/EDA en un directorio independiente.
