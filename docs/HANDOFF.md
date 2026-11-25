# Entrega y continuidad

La implementación inicial define experiment001–005. El runner ejecuta los experimentos y genera la entrega reproducible. Antes de declarar concluido el proyecto, comprobar el estado de Actions, revisar las tablas y figuras, confirmar las 3,300 observaciones, las siete familias supervisadas y tres ensambles, los cuatro algoritmos de agrupamiento y tres contrastes de Holm; inspeccionar el PDF y ampliar la interpretación sustantiva si procede. Los resultados definitivos son los archivos versionados o artefactos del runner, nunca una predicción de desempeño.

Si falla el runner, revisar etiquetas Windows/X64, disponibilidad de Conda, acceso a Figshare y permisos del token. Si la descarga falla, reintentar sin cambiar los conjuntos. Las siguientes investigaciones deben usar experiment006, etc. No reajustar basándose en la prueba ya observada.

La primera ejecución 45568371842 falló antes de las pruebas porque no existe pwsh en el runner. Se cambia a Windows PowerShell y se invoca conda.exe directamente. experiment005 está integrado en 003 para congelar el ranking de diez modelos antes de prueba. Revisar los nuevos tiempos y gráficos, sin inferir resultados hasta ejecutarlos.
