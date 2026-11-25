"""Selección con CV común, evaluación final y contrastes bootstrap pareados."""
from itertools import combinations
from time import perf_counter
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import parallel_config
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.multitest import multipletests
from .ensambles import construir
from .diagnosticos import diagnosticos
from .comun import carpeta, entrenamiento, vectorizador, cv, figura, guardar, SEMILLA, json_guardar

def modelos():
    return {
        "Regresión logística": (LogisticRegression(max_iter=1000, random_state=SEMILLA),
                                {"modelo__C": [.1, 1, 10]}, False),
        "SGD": (SGDClassifier(loss="log_loss", random_state=SEMILLA),
                {"modelo__alpha": [1e-5, 1e-4, 1e-3]}, False),
        "Árbol de decisión": (DecisionTreeClassifier(random_state=SEMILLA),
                             {"modelo__max_depth": [15, 30, None]}, False),
        "Random Forest": (RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=SEMILLA),
                          {"modelo__max_depth": [20, None], "modelo__max_features": ["sqrt", .1]}, False),
        "AdaBoost": (AdaBoostClassifier(random_state=SEMILLA),
                     {"modelo__n_estimators": [50, 100], "modelo__learning_rate": [.5, 1.]}, False),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=SEMILLA),
                              {"modelo__n_estimators": [50, 100], "modelo__max_depth": [1, 2]}, True),
        "SVM RBF": (SVC(kernel="rbf", random_state=SEMILLA),
                    {"modelo__C": [1, 10], "modelo__gamma": ["scale", .1]}, False),
    }

def pipeline(modelo, reducir=False):
    return Pipeline([("texto", vectorizador()),
                     ("svd", TruncatedSVD(50, random_state=SEMILLA) if reducir else "passthrough"),
                     ("modelo", modelo)])

def bootstrap(y, predicciones, B=2000):
    """Bootstrap estratificado, mismos índices para todos los modelos.
    p bilateral aproximado con distribución bootstrap centrada bajo H0.
    """
    y = np.asarray(y)
    nombres = list(predicciones)
    grupos = [np.flatnonzero(y == c) for c in np.unique(y)]
    rng = np.random.default_rng(SEMILLA)
    observado = np.array([f1_score(y, predicciones[n], average="macro") for n in nombres])
    replicas = np.empty((B, len(nombres)))
    for b in range(B):
        ind = np.concatenate([rng.choice(g, len(g), replace=True) for g in grupos])
        replicas[b] = [f1_score(y[ind], np.asarray(predicciones[n])[ind], average="macro") for n in nombres]
    filas = []
    for i, j in combinations(range(len(nombres)), 2):
        delta = observado[i] - observado[j]
        dist = replicas[:, i] - replicas[:, j]
        inferior, superior = np.quantile(dist, [.025, .975])
        p = (1 + np.count_nonzero(np.abs(dist-delta) >= abs(delta))) / (B+1)
        filas.append({"modelo_A": nombres[i], "modelo_B": nombres[j], "delta_F1": delta,
                      "IC_inferior": inferior, "IC_superior": superior, "p": p})
    tabla = pd.DataFrame(filas)
    tabla["rechaza_H0"], tabla["p_Holm"], _, _ = multipletests(tabla.p, alpha=.05, method="holm")
    return tabla, pd.DataFrame(replicas, columns=nombres)

def ejecutar(datos):
    ruta = carpeta(3)
    train, test = entrenamiento(datos), datos.loc[datos.particion.eq("test")]
    X, y = train.texto, train.categoria
    estimadores, validacion, detalles, pliegues = {}, [], [], []
    for nombre, (modelo, grid, reducir) in modelos().items():
        print(f"Ajustando {nombre}", flush=True)
        inicio = perf_counter()
        busqueda = GridSearchCV(pipeline(modelo, reducir), grid, cv=cv(), scoring="f1_macro",
                                n_jobs=-1, return_train_score=True, error_score="raise")
        with parallel_config(backend="loky", inner_max_num_threads=1):
            busqueda.fit(X, y)
        estimadores[nombre] = busqueda.best_estimator_
        validacion.append({"modelo": nombre, "F1_CV": busqueda.best_score_,
                           "DE_CV": busqueda.cv_results_["std_test_score"][busqueda.best_index_],
                           "parametros": str(busqueda.best_params_), "segundos": perf_counter()-inicio,
                           "segundos_refit": busqueda.refit_time_,
                           "segundos_ajuste_fold": busqueda.cv_results_["mean_fit_time"][busqueda.best_index_],
                           "familia": "Individual"})
        for i in range(5):
            pliegues.append({"modelo": nombre, "pliegue": i+1,
                "F1_validacion": busqueda.cv_results_[f"split{i}_test_score"][busqueda.best_index_],
                "F1_entrenamiento": busqueda.cv_results_[f"split{i}_train_score"][busqueda.best_index_]})
        detalle = pd.DataFrame(busqueda.cv_results_).assign(modelo=nombre)
        detalles.append(detalle)
    # experiment005: las bases usan configuraciones preespecificadas, no las ganadoras
    # de una búsqueda que hubiera visto las etiquetas de los pliegues externos.
    ruta_ens = carpeta(5)
    for nombre, ensamble in construir(modelos(), pipeline).items():
        print(f"Ajustando {nombre}", flush=True)
        inicio = perf_counter()
        with parallel_config(backend="loky", inner_max_num_threads=1):
            scores = cross_validate(ensamble, X, y, cv=cv(), scoring="f1_macro",
                                    n_jobs=-1, return_train_score=True, error_score="raise")
            t = perf_counter()
            ensamble.fit(X, y)
            refit = perf_counter()-t
        estimadores[nombre] = ensamble
        validacion.append({"modelo": nombre, "F1_CV": scores["test_score"].mean(),
                           "DE_CV": scores["test_score"].std(), "parametros": "Preespecificados; XGBoost CV interna",
                           "segundos": perf_counter()-inicio, "segundos_refit": refit,
                           "segundos_ajuste_fold": scores["fit_time"].mean(), "familia": "Ensamble"})
        for i, (v, e) in enumerate(zip(scores["test_score"], scores["train_score"])):
            pliegues.append({"modelo": nombre, "pliegue": i+1, "F1_validacion": v, "F1_entrenamiento": e})
    ranking = pd.DataFrame(validacion).sort_values("F1_CV", ascending=False).reset_index(drop=True)
    guardar(ranking, ruta, "validacion")
    guardar(pd.DataFrame(pliegues), ruta, "pliegues_cv")
    guardar(ranking.loc[ranking.familia.eq("Ensamble")], ruta_ens, "validacion_ensambles")
    guardar(pd.concat(detalles, ignore_index=True), ruta, "busqueda_cv")
    # Solo ahora se evalúa la prueba; ranking y top 3 congelados por CV.
    predicciones, filas = {}, []
    yt = test.categoria.to_numpy()
    clases = sorted(y.unique())
    for nombre in ranking.modelo:
        inicio = perf_counter()
        pred = estimadores[nombre].predict(test.texto)
        segundos_pred = perf_counter()-inicio
        predicciones[nombre] = pred
        filas.append({"modelo": nombre, "Accuracy": accuracy_score(yt, pred),
                      "Precision": precision_score(yt, pred, average="macro", zero_division=0),
                      "Recall": recall_score(yt, pred, average="macro", zero_division=0),
                      "F1": f1_score(yt, pred, average="macro"),
                      "segundos_prediccion": segundos_pred})
    metricas = pd.DataFrame(filas)
    guardar(metricas, ruta, "metricas_prueba")
    guardar(metricas.loc[metricas.modelo.isin(ranking.loc[ranking.familia.eq("Ensamble"), "modelo"])], ruta_ens, "metricas_ensambles")
    guardar(pd.DataFrame({"huella": test.huella.to_numpy(), "real": yt, **predicciones}), ruta, "predicciones")
    mejor = ranking.modelo.iloc[0]
    joblib.dump(estimadores[mejor], ruta / "modelo.joblib", compress=3)
    json_guardar({"seleccionado": mejor, "top3": ranking.modelo.head(3).tolist()}, ruta / "seleccion.json")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=ranking, y="modelo", x="F1_CV", ax=ax)
    ax.errorbar(ranking.F1_CV, np.arange(len(ranking)), xerr=ranking.DE_CV, fmt="none", color="black", capsize=3)
    ax.set(title="F1 macro en validación cruzada (media ± DE)", xlim=(0, 1), ylabel="")
    figura(ruta, "01_validacion", fig)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=metricas.melt("modelo", ["Accuracy", "Precision", "Recall", "F1"], var_name="metrica", value_name="valor"),
                y="modelo", x="valor", hue="metrica", ax=ax)
    ax.set(title="Evaluación final en prueba", xlim=(0, 1), ylabel="")
    figura(ruta, "02_metricas", fig)
    fig, axs = plt.subplots(3, 4, figsize=(19, 13))
    for ax, (nombre, pred) in zip(axs.flat, predicciones.items()):
        matriz = confusion_matrix(yt, pred, labels=clases)
        sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=clases, yticklabels=clases, ax=ax)
        ax.set_title(nombre)
    for ax in list(axs.flat)[len(predicciones):]:
        ax.axis("off")
    figura(ruta, "03_confusiones", fig)
    matriz = confusion_matrix(yt, predicciones[mejor], labels=clases)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", xticklabels=clases, yticklabels=clases, ax=ax)
    ax.set(title=f"Matriz de confusión: {mejor}", xlabel="Predicción", ylabel="Categoría real")
    figura(ruta, "04_mejor_confusion", fig)
    pares = pd.DataFrame(matriz, index=clases, columns=clases).rename_axis("real").reset_index().melt("real", var_name="predicha", value_name="errores")
    pares = pares.loc[pares.real.ne(pares.predicha)].sort_values("errores", ascending=False)
    guardar(pares, ruta, "errores_por_par")
    incorrectos = test.loc[yt != predicciones[mejor], ["huella", "categoria", "texto"]].copy()
    incorrectos["predicha"] = predicciones[mejor][yt != predicciones[mejor]]
    incorrectos["texto"] = incorrectos.texto.str.split().str[:30].str.join(" ")
    guardar(incorrectos, ruta, "ejemplos_errores")
    top = ranking.modelo.head(3)
    contrastes, replicas = bootstrap(yt, {n: predicciones[n] for n in top})
    guardar(contrastes, ruta, "bootstrap_holm")
    guardar(replicas, ruta, "replicas_bootstrap")
    etiquetas = contrastes.modelo_A + " − " + contrastes.modelo_B
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    ax = axs[0]
    pos = np.arange(len(contrastes))
    ax.hlines(pos, contrastes.IC_inferior, contrastes.IC_superior)
    ax.scatter(contrastes.delta_F1, pos)
    ax.axvline(0, ls="--", color="gray")
    ax.set(yticks=pos, yticklabels=etiquetas, xlabel="Diferencia F1 macro (IC 95% marginal)", title="Bootstrap pareado")
    ax = axs[1]
    sns.barplot(x=contrastes.p_Holm, y=etiquetas, ax=ax)
    ax.axvline(.05, ls="--", color="gray")
    ax.set(xlabel="p ajustado", ylabel="", title="Corrección de Holm", xlim=(0, 1))
    figura(ruta, "05_bootstrap_holm", fig)
    diagnosticos(ruta, ranking, metricas, pd.DataFrame(pliegues), yt, predicciones, clases, replicas, mejor)
