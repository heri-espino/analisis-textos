"""Agrupamiento independiente del clasificador, sin acceso a etiquetas durante el ajuste."""
from itertools import product
from time import perf_counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
from .comun import carpeta, vectorizador, figura, guardar, SEMILLA

def configuraciones(reducida):
    for k in [3, 4, 6, 8, 10]:
        yield "K-means", f"k={k}", KMeans(k, n_init=10, random_state=SEMILLA), False
        yield "Aglomerativo", f"k={k}", AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average"), True
        if reducida:
            yield "Mezcla gaussiana EM", f"k={k}", GaussianMixture(k, covariance_type="diag", n_init=2, random_state=SEMILLA), False
    for eps, minimo in product([.2, .35, .5, .65, .8], [5, 10]):
        yield "DBSCAN", f"eps={eps}, min={minimo}", DBSCAN(eps=eps, min_samples=minimo, metric="precomputed", n_jobs=1), True

def ajustar(nombre, parametros, modelo, precomputado, X, D, representacion):
    inicio = perf_counter()
    etiquetas = modelo.fit_predict(D if precomputado else X)
    tiempo_ajuste = perf_counter()-inicio
    inicio = perf_counter()
    valido = etiquetas != -1
    grupos = len(np.unique(etiquetas[valido]))
    cobertura = valido.mean()
    sil = (silhouette_score(D[np.ix_(valido, valido)], etiquetas[valido], metric="precomputed")
           if 2 <= grupos < valido.sum() else np.nan)
    # Penalización explícita por descartar documentos como ruido.
    criterio = cobertura * (1 + sil) / 2 if np.isfinite(sil) else np.nan
    return {"algoritmo": nombre, "parametros": parametros, "representacion": representacion,
            "grupos": grupos, "cobertura": cobertura, "silhouette": sil, "criterio": criterio,
            "segundos_ajuste": tiempo_ajuste, "segundos_silhouette": perf_counter()-inicio}, etiquetas

def ejecutar(datos):
    ruta = carpeta(2)
    # Las etiquetas reales no se consultan hasta terminar todas las configuraciones.
    v = vectorizador()
    X = v.fit_transform(datos.texto).astype(np.float32)
    svd = TruncatedSVD(50, random_state=SEMILLA)
    Z = normalize(svd.fit_transform(X))
    resultados = []
    for nombre, matriz in [("TF-IDF", X), ("SVD-50", Z)]:
        D = cosine_distances(matriz)
        np.clip(D, 0, 2, out=D)
        np.fill_diagonal(D, 0)
        resultados.extend(Parallel(n_jobs=-1, prefer="threads")(
            delayed(ajustar)(a, p, m, pre, matriz, D, nombre)
            for a, p, m, pre in configuraciones(nombre == "SVD-50")))
    tabla = pd.DataFrame([fila for fila, _ in resultados])
    ganador = tabla.criterio.idxmax()
    # Evaluación externa exclusivamente posterior a la selección interna.
    y = datos.categoria
    tabla["ARI"] = [adjusted_rand_score(y, lab) for _, lab in resultados]
    tabla["NMI"] = [normalized_mutual_info_score(y, lab) for _, lab in resultados]
    tabla["seleccionado"] = tabla.index == ganador
    guardar(tabla, ruta, "configuraciones")
    seleccion = tabla.dropna(subset=["criterio"]).sort_values("criterio", ascending=False)
    mejores = seleccion.groupby(["algoritmo", "representacion"], sort=False).head(1)
    guardar(mejores, ruta, "mejores")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=mejores, y="algoritmo", x="silhouette", hue="representacion", ax=ax)
    ax.set(title="Mejores configuraciones por criterio interno", ylabel="")
    figura(ruta, "01_silhouette", fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=mejores, y="algoritmo", x="ARI", hue="representacion", ax=ax)
    ax.set(title="Concordancia externa posterior: ARI", ylabel="")
    figura(ruta, "02_ari", fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(data=tabla, x="silhouette", y="NMI", hue="algoritmo", style="representacion", size="cobertura", ax=ax)
    ax.set(title="Calidad geométrica frente a concordancia temática")
    figura(ruta, "03_concordancia", fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(data=tabla, x="grupos", y="criterio", hue="algoritmo", style="representacion", ax=ax)
    ax.set(title="Número de grupos y criterio interno")
    figura(ruta, "04_numero_grupos", fig)
    lab = resultados[ganador][1]
    guardar(pd.DataFrame({"huella": datos.huella, "categoria": y, "grupo": lab}), ruta, "asignaciones")
    mapa = pd.DataFrame({"Componente 1": Z[:, 0], "Componente 2": Z[:, 1],
                         "Categoría": y, "Grupo": lab.astype(str)})
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for ax, hue in zip(axs, ["Grupo", "Categoría"]):
        sns.scatterplot(data=mapa, x="Componente 1", y="Componente 2", hue=hue, s=12, alpha=.65, ax=ax)
        ax.set_title(hue)
    figura(ruta, "05_proyeccion", fig)
    cruzada = pd.crosstab(y, lab, normalize="index")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(cruzada, annot=True, fmt=".2f", cmap="Blues", ax=ax)
    ax.set(title="Categorías originales frente al agrupamiento seleccionado", xlabel="Grupo", ylabel="Categoría")
    figura(ruta, "06_cruzada", fig)
    filas = []
    for grupo in np.unique(lab):
        indices = np.flatnonzero(lab == grupo)
        centro = np.asarray(X[indices].mean(axis=0)).ravel()
        palabras = v.get_feature_names_out()[centro.argsort()[-12:][::-1]]
        # Ejemplos cercanos al centro temático, sin cabeceras ni correos.
        cercanos = np.asarray(X[indices] @ centro).ravel().argsort()[-2:][::-1]
        ejemplos = [" ".join(datos.texto.iloc[indices[i]].split()[:30]) for i in cercanos]
        filas.append({"grupo": int(grupo), "documentos": len(indices), "palabras": ", ".join(palabras),
                      "ejemplo_1": ejemplos[0], "ejemplo_2": ejemplos[-1]})
    guardar(pd.DataFrame(filas), ruta, "interpretacion")

    fig, ax = plt.subplots(figsize=(10,5))
    sns.boxplot(data=tabla, y="algoritmo", x="segundos_ajuste", hue="representacion", ax=ax)
    ax.set(title="Costo de ajuste por configuración (con ejecución paralela)", xlabel="Segundos de pared", ylabel="", xscale="log")
    figura(ruta, "07_tiempos", fig)
    fig, axs = plt.subplots(1,2,figsize=(12,5))
    for ax, (rep, sub) in zip(axs, tabla.loc[tabla.algoritmo.eq("DBSCAN")].groupby("representacion")):
        sub = sub.assign(eps=sub.parametros.str.extract(r"eps=([^,]+)")[0].astype(float),
                         minimo=sub.parametros.str.extract(r"min=(\d+)")[0].astype(int))
        sns.heatmap(sub.pivot(index="minimo", columns="eps", values="cobertura"), annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=ax)
        ax.set_title(rep)
    fig.suptitle("DBSCAN: proporción de documentos no clasificados como ruido")
    figura(ruta, "08_cobertura_dbscan", fig)
