"""EDA y comparación de representaciones; vocabulario ajustado solo en entrenamiento."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from .comun import carpeta, entrenamiento, vectorizador, figura, guardar, SEMILLA

def ejecutar(datos):
    ruta = carpeta(1)
    train = entrenamiento(datos).copy()
    guardar(datos.drop(columns="texto"), ruta, "particiones")
    train["palabras"] = train.texto.str.split().str.len()
    train["caracteres"] = train.texto.str.len()
    resumen = train.groupby("categoria").agg(
        documentos=("texto", "size"), mediana_palabras=("palabras", "median"),
        media_palabras=("palabras", "mean"), mediana_caracteres=("caracteres", "median")).reset_index()
    guardar(resumen, ruta, "eda")
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.countplot(data=train, y="categoria", ax=ax)
    ax.set(title="Documentos de entrenamiento por categoría", xlabel="Documentos", ylabel="")
    figura(ruta, "01_categorias", fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.boxplot(data=train, y="categoria", x="palabras", showfliers=False, ax=ax)
    ax.set(title="Longitud de los documentos (sin extremos visibles)", xlabel="Palabras", ylabel="")
    figura(ruta, "02_longitudes", fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(np.log1p(train.palabras), bins=35, ax=ax)
    ax.set(title="Distribución de longitud", xlabel="log(1 + palabras)", ylabel="Documentos")
    figura(ruta, "03_histograma", fig)
    filas = []
    for tipo in ["BoW", "TF-IDF"]:
        v = vectorizador(tipo)
        X = v.fit_transform(train.texto)
        pesos = np.asarray(X.sum(axis=0)).ravel()
        terminos = pd.DataFrame({"palabra": v.get_feature_names_out(), "peso": pesos})
        terminos = terminos.nlargest(20, "peso")
        guardar(terminos, ruta, f"terminos_{tipo}")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=terminos, y="palabra", x="peso", ax=ax)
        ax.set(title=f"20 palabras con mayor peso: {tipo}", xlabel="Peso acumulado", ylabel="")
        figura(ruta, f"04_terminos_{tipo}", fig)
        filas.append({"representacion": tipo, "documentos": X.shape[0],
                      "caracteristicas": X.shape[1], "no_ceros": X.nnz,
                      "esparsidad": 1-X.nnz/(X.shape[0]*X.shape[1])})
        if tipo == "TF-IDF":
            svd = TruncatedSVD(100, random_state=SEMILLA).fit(X)
            var = pd.DataFrame({"componentes": np.arange(1, 101),
                                "varianza_acumulada": svd.explained_variance_ratio_.cumsum()})
            guardar(var, ruta, "varianza_svd")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.lineplot(data=var, x="componentes", y="varianza_acumulada", ax=ax)
            ax.set(title="Varianza explicada acumulada por SVD", ylabel="Proporción acumulada")
            figura(ruta, "05_varianza_svd", fig)
    tabla = pd.DataFrame(filas)
    guardar(tabla, ruta, "representaciones")
    fig, ax = plt.subplots(figsize=(7, 3))
    sns.barplot(data=tabla, x="representacion", y="esparsidad", ax=ax)
    ax.set(title="Esparsidad de las matrices", ylim=(0, 1), xlabel="", ylabel="Proporción de ceros")
    figura(ruta, "06_esparsidad", fig)
