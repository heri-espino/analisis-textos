"""Visualizaciones que relacionan precisión, costo, estabilidad y diversidad."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score
from .comun import figura, guardar

def diagnosticos(ruta, ranking, metricas, pliegues, y, predicciones, clases, replicas, mejor):
    orden = ranking.modelo.tolist()
    datos = ranking.merge(metricas, on="modelo")
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    sns.barplot(data=datos, y="modelo", x="segundos", order=orden, hue="familia", dodge=False, ax=axs[0])
    sns.barplot(data=datos, y="modelo", x="segundos_prediccion", order=orden, hue="familia", dodge=False, ax=axs[1])
    axs[0].set(title="Búsqueda CV y ajuste final", xlabel="Segundos de pared", ylabel="", xscale="log")
    axs[1].set(title="Predicción de todos los documentos de prueba", xlabel="Segundos de pared", ylabel="", xscale="log")
    figura(ruta, "06_tiempos", fig)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(data=pliegues, y="modelo", x="F1_validacion", order=orden, ax=ax)
    sns.stripplot(data=pliegues, y="modelo", x="F1_validacion", order=orden, color="black", ax=ax)
    ax.set(title="Estabilidad de los modelos en los cinco pliegues", xlabel="F1 macro de validación", ylabel="")
    figura(ruta, "07_estabilidad_cv", fig)
    fig, ax = plt.subplots(figsize=(11, 6))
    promedios = pliegues.groupby("modelo")[["F1_entrenamiento","F1_validacion"]].mean().reset_index()
    sns.barplot(data=promedios.melt("modelo", var_name="particion", value_name="F1"),
                y="modelo", x="F1", hue="particion", order=orden, ax=ax)
    ax.set(title="Brecha entre entrenamiento y validación", ylabel="", xlim=(0,1))
    figura(ruta, "08_brecha_cv", fig)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.scatterplot(data=datos, x="segundos", y="F1_CV", hue="familia", s=90, ax=ax)
    for fila in datos.itertuples():
        ax.annotate(fila.modelo, (fila.segundos, fila.F1_CV), xytext=(4,4), textcoords="offset points", fontsize=8)
    ax.set(xscale="log", title="Compromiso entre rendimiento y costo", xlabel="Segundos de búsqueda y ajuste (log)", ylabel="F1 macro CV")
    ax.margins(x=.3, y=.15)
    figura(ruta, "09_costo_rendimiento", fig)
    por_clase = pd.DataFrame({n: f1_score(y, pred, labels=clases, average=None, zero_division=0)
                             for n, pred in predicciones.items()}, index=clases).T
    guardar(por_clase.rename_axis("modelo").reset_index(), ruta, "f1_por_clase")
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(por_clase.loc[orden], annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=ax)
    ax.set(title="¿Qué categorías reconoce cada modelo?", xlabel="Categoría real", ylabel="")
    figura(ruta, "10_f1_por_clase", fig)
    # Diversidad: desacuerdo de predicciones y coincidencia de errores en prueba.
    P = np.column_stack([predicciones[n] for n in orden])
    E = P != np.asarray(y)[:,None]
    desacuerdo = pd.DataFrame((P[:,:,None] != P[:,None,:]).mean(axis=0), index=orden, columns=orden)
    errores = pd.DataFrame(E.astype(float), columns=orden).corr()
    guardar(desacuerdo.rename_axis("modelo").reset_index(), ruta, "desacuerdo")
    guardar(errores.rename_axis("modelo").reset_index(), ruta, "correlacion_errores")
    fig, axs = plt.subplots(1,2,figsize=(18,7))
    sns.heatmap(desacuerdo, annot=True, fmt=".2f", cmap="Blues", ax=axs[0])
    sns.heatmap(errores, annot=True, fmt=".2f", cmap="vlag", center=0, ax=axs[1])
    axs[0].set_title("Proporción de predicciones distintas")
    axs[1].set_title("Correlación de indicadores de error")
    figura(ruta, "11_diversidad", fig)
    fig, ax = plt.subplots(figsize=(10,5))
    sns.boxplot(data=replicas.melt(var_name="modelo", value_name="F1"), y="modelo", x="F1", ax=ax)
    ax.set(title="Distribuciones bootstrap de los tres finalistas", ylabel="", xlabel="F1 macro")
    figura(ruta, "12_bootstrap_distribuciones", fig)
    referencia = ranking.loc[ranking.familia.eq("Individual")].iloc[0].modelo
    base_cv = datos.loc[datos.modelo.eq(referencia), "F1_CV"].iloc[0]
    base_test = datos.loc[datos.modelo.eq(referencia), "F1"].iloc[0]
    comparacion = datos.loc[datos.familia.eq("Ensamble")].copy()
    comparacion["delta_CV"] = comparacion.F1_CV-base_cv
    comparacion["delta_prueba"] = comparacion.F1-base_test
    guardar(comparacion[["modelo","delta_CV","delta_prueba"]], ruta, "mejora_ensambles")
    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(data=comparacion.melt("modelo", ["delta_CV","delta_prueba"], var_name="particion", value_name="delta"),
                y="modelo", x="delta", hue="particion", ax=ax)
    ax.axvline(0, color="gray", ls="--")
    ax.set(title=f"Ganancia de los ensambles frente a {referencia}", xlabel="Diferencia en F1 macro", ylabel="")
    figura(ruta, "13_mejora_ensambles", fig)
