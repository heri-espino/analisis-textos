"""Tres estudios controlados mediante los mismos pliegues de entrenamiento."""
from time import perf_counter
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from joblib import parallel_config
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from .comun import carpeta, entrenamiento, vectorizador, cv, figura, guardar, SEMILLA

def evaluar(datos, grid):
    train = entrenamiento(datos)
    pipe = Pipeline([("texto", vectorizador()), ("svd", "passthrough"),
                     ("modelo", LogisticRegression(max_iter=1000, random_state=SEMILLA))])
    busqueda = GridSearchCV(pipe, grid, cv=cv(), scoring="f1_macro", n_jobs=-1,
                            refit=False, return_train_score=True, error_score="raise")
    inicio = perf_counter()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        busqueda.fit(train.texto, train.categoria)
    r = pd.DataFrame(busqueda.cv_results_)
    return r.assign(segundos_busqueda=perf_counter()-inicio, error_entrenamiento=1-r.mean_train_score,
                    error_validacion=1-r.mean_test_score)

def ejecutar(datos):
    ruta = carpeta(4)
    r = evaluar(datos, {"texto": [vectorizador("BoW"), vectorizador("TF-IDF")]})
    r["representacion"] = ["BoW", "TF-IDF"]
    guardar(r, ruta, "representacion")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=r, x="representacion", y="mean_test_score", ax=ax)
    ax.set(title="Efecto de la representación (C=1)", ylabel="F1 macro de CV", ylim=(0, 1))
    figura(ruta, "01_representacion", fig)
    r = evaluar(datos, {"texto__max_features": [1000, 3000, 10000, 20000]})
    r["caracteristicas_max"] = r.param_texto__max_features.astype(int)
    guardar(r, ruta, "dimensionalidad")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=r, x="caracteristicas_max", y="mean_test_score", marker="o", ax=ax)
    ax.set(title="Efecto del máximo de características", ylabel="F1 macro de CV", xlabel="Máximo de características")
    figura(ruta, "02_dimensionalidad", fig)
    r = evaluar(datos, {"svd": [TruncatedSVD(k, random_state=SEMILLA) for k in [25, 50, 100, 200]]})
    r["componentes"] = [25, 50, 100, 200]
    guardar(r, ruta, "componentes_svd")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=r, x="componentes", y="mean_test_score", marker="o", ax=ax)
    ax.set(title="Efecto de la dimensión latente SVD", ylabel="F1 macro de CV")
    figura(ruta, "03_componentes_svd", fig)
    r = evaluar(datos, {"modelo__C": [.001, .01, .1, 1, 10, 100, 1000]})
    r["C"] = r.param_modelo__C.astype(float)
    guardar(r, ruta, "complejidad")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=r.melt("C", ["error_entrenamiento", "error_validacion"], var_name="particion", value_name="error"),
                 x="C", y="error", hue="particion", marker="o", ax=ax)
    ax.set(xscale="log", title="Regularización y brecha de generalización", ylabel="Error = 1 − F1 macro")
    figura(ruta, "04_complejidad", fig)
