"""Diagnósticos complementarios: conservar protocolos y resultados originales."""
from time import perf_counter
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import seaborn as sns
from .comun import carpeta, vectorizador, guardar, figura, SEMILLA
from .generalizacion import evaluar

def ejecutar(datos):
    ruta = carpeta(8)
    previo = pd.read_csv(carpeta(2) / "asignaciones.csv")
    assert datos.huella.is_unique and previo.huella.is_unique
    lab = datos[["huella"]].merge(previo[["huella", "grupo"]], on="huella",
                                    how="left", validate="one_to_one").grupo
    assert lab.notna().all() and len(lab) == len(previo) == 3300
    v = vectorizador()
    X = v.fit_transform(datos.texto).astype(np.float32)
    Z = normalize(TruncatedSVD(50, random_state=SEMILLA).fit_transform(X))
    filas, arboles = [], {}
    for rep, A in [("TF-IDF", X), ("SVD-50", Z)]:
        D = cosine_distances(A)
        np.clip(D, 0, 2, out=D)
        D = (D + D.T) / 2
        np.fill_diagonal(D, 0)
        for k in [3, 4, 6, 8, 10]:
            inicio = perf_counter()
            modelo = KMeans(k, n_init=10, random_state=SEMILLA).fit(A)
            filas.append({"representacion": rep, "k": k, "inercia": modelo.inertia_,
                          "silhouette": silhouette_score(D, modelo.labels_, metric="precomputed"),
                          "segundos": perf_counter()-inicio,
                          "ARI_grupos_guardados": adjusted_rand_score(lab, modelo.labels_)})
        # Enlace promedio con exactamente las distancias coseno del análisis original.
        L = linkage(squareform(D, checks=False), method="average")
        arboles[rep] = L
        guardar(pd.DataFrame(L, columns=["izquierda","derecha","distancia","n"]),
                ruta, "enlace_" + rep)
        corte = cut_tree(L, n_clusters=3).ravel()
        guardar(pd.DataFrame({"huella": datos.huella, "grupo": corte}),
                ruta, "aglomerativo_" + rep)
    tabla = pd.DataFrame(filas)
    guardar(tabla, ruta, "kmeans")
    fig, axs = plt.subplots(1,2,figsize=(10,4))
    for ax, y in zip(axs, ["inercia", "silhouette"]):
        sns.lineplot(data=tabla, x="k", y=y, hue="representacion", marker="o", ax=ax)
    figura(ruta, "01_kmeans", fig)
    for rep, L in arboles.items():
        fig, ax = plt.subplots(figsize=(10,4))
        dendrogram(L, truncate_mode="lastp", p=25, show_leaf_counts=True,
                   color_threshold=0, above_threshold_color="#203e51", ax=ax)
        ax.set(xlabel="Subárboles terminales (tamaño entre paréntesis)",
               ylabel="Distancia coseno media", title=rep)
        figura(ruta, "02_dendrograma_" + rep, fig)
    # Los términos se ordenan por peso medio TF-IDF, NO por frecuencia ni por un peso simulado.
    palabras = v.get_feature_names_out()
    pesos = []
    for g in sorted(lab.unique()):
        centro = np.asarray(X[lab.to_numpy() == g].mean(axis=0)).ravel()
        for rango, j in enumerate(centro.argsort()[-12:][::-1], 1):
            pesos.append({"grupo": int(g), "rango": rango, "palabra": palabras[j],
                          "peso_medio_tfidf": float(centro[j])})
    guardar(pd.DataFrame(pesos), ruta, "pesos_terminos")
    # Repetir el estudio SVD en los mismos cinco pliegues; ni ajuste ni selección en prueba.
    ks = [25,50,100,200,300,400,500,750,1000]
    scores = evaluar(datos, {"svd": [TruncatedSVD(k, random_state=SEMILLA) for k in ks]})
    scores["componentes"] = ks
    guardar(scores, ruta, "componentes_svd")
    fig, ax = plt.subplots(figsize=(7,4))
    sns.lineplot(data=scores, x="componentes", y="mean_test_score", marker="o", ax=ax)
    ax.set(ylabel="F1 macro CV", xlabel="Componentes SVD")
    figura(ruta, "03_svd", fig)
