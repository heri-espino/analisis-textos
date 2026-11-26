"""experiment007: seis vistas bidimensionales de una misma representación."""
from time import perf_counter
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding, trustworthiness
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import connected_components
from .comun import SEMILLA, carpeta, vectorizador, guardar, json_guardar

METODOS = ["PCA", "ICA", "t-SNE", "UMAP", "Isomap", "Spectral Embedding"]

def proyectar(X):
    from umap import UMAP

    # Entrada idéntica para los seis métodos; nunca se entregan etiquetas.
    inicio = perf_counter()
    Z = TruncatedSVD(100, random_state=SEMILLA).fit_transform(X)
    tiempo_svd = perf_counter() - inicio
    grafo = kneighbors_graph(Z, 15, mode="connectivity", n_jobs=-1)
    componentes = connected_components(grafo, directed=False, return_labels=False)
    modelos = {
        "PCA": PCA(n_components=2, svd_solver="full"),
        "ICA": FastICA(n_components=2, random_state=SEMILLA, max_iter=2000,
                       tol=1e-3, whiten="unit-variance"),
        "t-SNE": TSNE(n_components=2, random_state=SEMILLA, init="pca",
                      learning_rate="auto", perplexity=30, n_jobs=-1),
        "UMAP": UMAP(n_components=2, random_state=SEMILLA, n_neighbors=15,
                     min_dist=.1, metric="euclidean", n_jobs=1),
        "Isomap": Isomap(n_components=2, n_neighbors=15, n_jobs=-1),
        "Spectral Embedding": SpectralEmbedding(n_components=2,
                      random_state=SEMILLA, n_neighbors=15, n_jobs=-1),
    }
    coordenadas, filas, avisos = {}, [], {}
    for nombre, modelo in modelos.items():
        print(f"Proyectando {nombre}", flush=True)
        inicio = perf_counter()
        with warnings.catch_warnings(record=True) as registrados:
            warnings.simplefilter("always")
            Y = modelo.fit_transform(Z)
        tiempo = perf_counter()-inicio
        if Y.shape != (len(Z), 2) or not np.isfinite(Y).all():
            raise ValueError(f"Coordenadas inválidas: {nombre}")
        coordenadas[nombre] = Y
        avisos[nombre] = list(dict.fromkeys(str(a.message) for a in registrados))
        filas.append({"metodo": nombre, "documentos": len(Z),
                      "segundos_proyeccion": tiempo,
                      "trustworthiness_15": trustworthiness(Z, Y, n_neighbors=15)})
    metadata = {"componentes_svd": 100, "segundos_svd": tiempo_svd,
                "componentes_grafo_15": int(componentes),
                "semilla": SEMILLA, "avisos": avisos,
                "varianza_pca_2d": float(modelos["PCA"].explained_variance_ratio_.sum()),
                "parametros": {n: m.get_params() for n,m in modelos.items()}}
    return coordenadas, pd.DataFrame(filas), metadata

def alinear_grupos(datos, asignaciones):
    """Recuperar el agrupamiento ya seleccionado por huella, no por orden de filas."""
    if not datos.huella.is_unique or not asignaciones.huella.is_unique:
        raise ValueError("Huellas duplicadas")
    if set(datos.huella) != set(asignaciones.huella):
        raise ValueError("La muestra no coincide con experiment002")
    grupos = asignaciones.set_index("huella").loc[datos.huella]
    if not np.array_equal(grupos.categoria.to_numpy(), datos.categoria.to_numpy()):
        raise ValueError("Las categorías no coinciden")
    return grupos.grupo.to_numpy()

def tabla_coordenadas(coordenadas, datos, grupos):
    return pd.concat([
        pd.DataFrame({"metodo": nombre, "huella": datos.huella.to_numpy(),
                      "x": Y[:,0], "y": Y[:,1],
                      "categoria": datos.categoria.to_numpy(), "grupo": grupos})
        for nombre, Y in coordenadas.items()
    ], ignore_index=True)

def dashboard(tabla, ruta, variable, nombre, titulo):
    # Colores y orden de puntos idénticos en los seis paneles.
    tabla = tabla.copy()
    if variable == "grupo":
        tabla[variable] = tabla[variable].map(lambda g: "Ruido" if g == -1 else f"Grupo {g}")
        orden = [f"Grupo {g}" for g in sorted(tabla.grupo.str.replace("Grupo ", "", regex=False)
                    .loc[tabla.grupo.ne("Ruido")].astype(int).unique())]
        if "Ruido" in tabla.grupo.values:
            orden.append("Ruido")
    else:
        orden = sorted(tabla[variable].unique())
    paleta = dict(zip(orden, sns.color_palette("dark", n_colors=len(orden))))
    if "Ruido" in paleta:
        paleta["Ruido"] = (.55,.55,.55)
    fig, axs = plt.subplots(2,3,figsize=(15,9))
    for ax, metodo in zip(axs.flat, METODOS):
        sub = tabla.loc[tabla.metodo.eq(metodo)]
        sns.scatterplot(data=sub, x="x", y="y", hue=variable, hue_order=orden,
                        palette=paleta, s=11, alpha=.64, linewidth=0,
                        legend=False, ax=ax)
        ax.set(title=metodo, xlabel="", ylabel="", xticks=[], yticks=[])
        ax.set_aspect("equal", adjustable="datalim")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", linestyle="none", color=paleta[g],
                      markersize=6, label=g) for g in orden]
    fig.legend(handles=handles, loc="lower center", ncol=min(6,len(orden)),
               frameon=False, title="Categoría real" if variable=="categoria" else "Grupo descubierto")
    fig.suptitle(titulo, fontsize=17, y=.98)
    fig.tight_layout(rect=(0,.10,1,.94))
    fig.savefig(ruta / f"{nombre}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

def ejecutar(datos):
    ruta = carpeta(7)
    asignaciones = pd.read_csv(carpeta(2) / "asignaciones.csv")
    grupos = alinear_grupos(datos, asignaciones)
    X = vectorizador().fit_transform(datos.texto).astype(np.float32)
    coords, metricas, metadata = proyectar(X)
    tabla = tabla_coordenadas(coords, datos, grupos)
    guardar(tabla, ruta, "proyecciones_2d")
    guardar(metricas, ruta, "metricas_proyecciones")
    # Parámetros simples, serializables; no almacenar objetos de Python.
    metadata["parametros"] = {n: {k: str(v) for k,v in p.items()}
                              for n,p in metadata["parametros"].items()}
    json_guardar(metadata, ruta / "metadatos.json")
    dashboard(tabla, ruta, "categoria", "01_dashboard_categorias",
              "Seis proyecciones del corpus por categoría real")
    dashboard(tabla, ruta, "grupo", "02_dashboard_grupos",
              "Seis proyecciones del corpus por grupo descubierto")
