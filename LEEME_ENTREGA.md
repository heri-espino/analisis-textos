# Entrega rápida LDS1071

1. Descomprimir el ZIP completo.
2. Abrir notebook_resultados.ipynb en Jupyter desde esta carpeta.
3. Usar Python 3.12 y el entorno original, o instalar requirements_resultados.txt.
4. Ejecutar todas las celdas. No se descarga el corpus ni se ajustan modelos.

La carpeta outputs contiene los CSV, PNG y los dos pipelines Joblib históricos. Las predicciones de los diez candidatos originales y diez híbridos permiten verificar las métricas; únicamente los mejores pipelines de cada familia de representación se serializaron. Para repetir los ajustes desde cero, consultar notebook_autonoma.ipynb en el repositorio. En Colab, subir/descomprimir el ZIP, abrir la notebook y ejecutar; si se reinstalan bibliotecas, reiniciar el kernel. No abrir modelos Joblib de procedencia desconocida.
