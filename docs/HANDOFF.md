# Entrega y continuidad

La implementación inicial define experiment001–005. El runner ejecuta los experimentos y genera la entrega reproducible. Antes de declarar concluido el proyecto, comprobar el estado de Actions, revisar las tablas y figuras, confirmar las 3,300 observaciones, las siete familias supervisadas y tres ensambles, los cuatro algoritmos de agrupamiento y tres contrastes de Holm; inspeccionar el PDF y ampliar la interpretación sustantiva si procede. Los resultados definitivos son los archivos versionados o artefactos del runner, nunca una predicción de desempeño.

Si falla el runner, revisar etiquetas Windows/X64, disponibilidad de Conda, acceso a Figshare y permisos del token. Si la descarga falla, reintentar sin cambiar los conjuntos. Las siguientes investigaciones deben usar experiment006, etc. No reajustar basándose en la prueba ya observada.

La primera ejecución 45568371842 falló antes de las pruebas porque no existe pwsh en el runner. Se cambia a Windows PowerShell y se invoca conda.exe directamente. experiment005 está integrado en 003 para congelar el ranking de diez modelos antes de prueba. Revisar los nuevos tiempos y gráficos, sin inferir resultados hasta ejecutarlos.

La segunda ejecución 45569130565 no generó jobs: corregir la sintaxis YAML de los comandos PowerShell con bloques literales. Las métricas de clasificación se grafican separadas de los tiempos.

La tercera ejecución 45569178773 falló por instalación existente en miniconda3. Se reutiliza la instalación local, se detecta conda.exe y se ejecuta conda env update -n analisis-textos -f environment.yml --prune.

experiment006: conservar la primera ejecución exitosa en c1324670b7bbbf15d9e97c7b1b181c9d7ecb227f. Repetir diez candidatos con 20 indicadores temáticos; comparar por algoritmo y construir ranking global. La prueba ya observada es exploratoria. Validar los resultados nuevos y el PDF antes de entregar. Siguiente investigación: experiment007.
