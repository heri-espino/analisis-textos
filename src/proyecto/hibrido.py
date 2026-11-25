"""experiment006: ablación apareada de características temáticas."""
from time import perf_counter
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import confusion_matrix
from .clasificacion import ajustar_modelos, evaluar_modelos, bootstrap
from .comun import carpeta, entrenamiento, guardar, figura, json_guardar

def ejecutar(datos):
    base, ruta = carpeta(3), carpeta(6)
    train, test = entrenamiento(datos), datos.loc[datos.particion.eq("test")]
    # Comprobar que la comparación conserva exactamente documentos y orden.
    anteriores = pd.read_csv(base / "predicciones.csv").set_index("huella")
    assert anteriores.index.is_unique and set(anteriores.index) == set(test.huella)
    anteriores = anteriores.loc[test.huella]
    yt = test.categoria.to_numpy()
    assert np.array_equal(anteriores.real.to_numpy(), yt)
    vb = pd.read_csv(base / "validacion.csv")
    mb = pd.read_csv(base / "metricas_prueba.csv")
    pb = pd.read_csv(base / "pliegues_cv.csv")
    estimadores, vh, ph, detalles = ajustar_modelos(train.texto, train.categoria, con_grupos=True)
    assert set(vb.modelo) == set(vh.modelo)
    # Selección híbrida congelada antes de evaluar predicciones híbridas.
    guardar(vh, ruta, "validacion_hibrida")
    guardar(ph, ruta, "pliegues_hibridos")
    guardar(detalles, ruta, "busqueda_cv")
    pred, mh = evaluar_modelos(estimadores, vh, test)
    guardar(mh, ruta, "metricas_hibridas")
    guardar(pd.DataFrame({"huella": test.huella.to_numpy(), "real": yt, **pred}), ruta, "predicciones_hibridas")
    joblib.dump(estimadores[vh.modelo.iloc[0]], ruta / "modelo.joblib", compress=3)
    def unir(b, h):
        return pd.concat([b.assign(representacion="Léxica"), h.assign(representacion="Léxica + grupos")], ignore_index=True)
    val, met, folds = unir(vb,vh), unir(mb,mh), unir(pb,ph)
    ranking = val.merge(met, on=["modelo","representacion"]).sort_values("F1_CV", ascending=False).reset_index(drop=True)
    guardar(ranking, ruta, "ranking_global")
    guardar(folds, ruta, "pliegues_comparados")
    pares = ranking.pivot(index="modelo",columns="representacion",values=["F1_CV","F1","segundos"])
    comparacion = pd.DataFrame({
        "modelo": pares.index,
        "F1_CV_base": pares["F1_CV"]["Léxica"].to_numpy(),
        "F1_CV_hibrido": pares["F1_CV"]["Léxica + grupos"].to_numpy(),
        "F1_prueba_base": pares["F1"]["Léxica"].to_numpy(),
        "F1_prueba_hibrido": pares["F1"]["Léxica + grupos"].to_numpy(),
        "segundos_base": pares["segundos"]["Léxica"].to_numpy(),
        "segundos_hibrido": pares["segundos"]["Léxica + grupos"].to_numpy(),
    })
    comparacion["delta_CV"] = comparacion.F1_CV_hibrido-comparacion.F1_CV_base
    comparacion["delta_prueba"] = comparacion.F1_prueba_hibrido-comparacion.F1_prueba_base
    comparacion["factor_tiempo"] = comparacion.segundos_hibrido/comparacion.segundos_base
    comparacion = comparacion.sort_values("delta_CV", ascending=False)
    guardar(comparacion, ruta, "comparacion")
    # Diez contrastes predefinidos: cada algoritmo contra su propia referencia.
    # Mismas semillas y mismos índices estratificados para todas las parejas.
    filas = []
    for nombre in vh.modelo:
        tabla, _ = bootstrap(yt, {"Híbrido": pred[nombre], "Base": anteriores[nombre].to_numpy()})
        filas.append(tabla.assign(modelo=nombre).drop(columns=["modelo_A","modelo_B","p_Holm","rechaza_H0"]))
    contrastes = pd.concat(filas, ignore_index=True)
    contrastes["rechaza_H0"], contrastes["p_Holm"], _, _ = multipletests(contrastes.p, alpha=.05, method="holm")
    contrastes = contrastes.sort_values("delta_F1", ascending=False)
    guardar(contrastes, ruta, "bootstrap_holm")
    # Ranking global y sus tres finalistas (familia inferencial separada, exploratoria).
    def etiqueta(fila):
        return fila.modelo + (" + grupos" if fila.representacion != "Léxica" else "")
    top = ranking.head(3)
    finalistas = {etiqueta(f): (pred[f.modelo] if f.representacion != "Léxica"
                              else anteriores[f.modelo].to_numpy()) for f in top.itertuples()}
    global_holm, replicas = bootstrap(yt, finalistas)
    guardar(global_holm, ruta, "bootstrap_finalistas")
    guardar(replicas, ruta, "replicas_finalistas")
    json_guardar({"modelo": ranking.iloc[0].modelo, "representacion": ranking.iloc[0].representacion,
                  "top3": list(finalistas), "evaluacion": "exploratoria; prueba observada antes de experiment006"},
                 ruta / "seleccion_global.json")
    orden = vb.modelo.tolist()
    fig, axs = plt.subplots(1,2,figsize=(15,6))
    for ax, tabla, metrica, titulo in [(axs[0],val,"F1_CV","Validación cruzada"),(axs[1],met,"F1","Prueba histórica")]:
        sns.barplot(data=tabla,y="modelo",x=metrica,hue="representacion",order=orden,ax=ax)
        ax.set(title=titulo,xlabel="F1 macro",ylabel="",xlim=(0,1))
    figura(ruta,"01_comparacion_pareada",fig)
    fig, ax = plt.subplots(figsize=(11,6))
    sns.barplot(data=comparacion.melt("modelo",["delta_CV","delta_prueba"],var_name="particion",value_name="delta"),
                y="modelo",x="delta",hue="particion",ax=ax)
    ax.axvline(0,ls="--",color="gray")
    ax.set(title="Ganancia atribuible a las características temáticas",xlabel="Diferencia F1 (híbrido − base)",ylabel="")
    figura(ruta,"02_ganancia",fig)
    dif = folds.pivot(index=["modelo","pliegue"],columns="representacion",values="F1_validacion").reset_index()
    dif["delta"] = dif["Léxica + grupos"]-dif["Léxica"]
    guardar(dif,ruta,"ganancia_pliegues")
    fig,ax=plt.subplots(figsize=(10,6))
    sns.boxplot(data=dif,y="modelo",x="delta",order=orden,ax=ax)
    sns.stripplot(data=dif,y="modelo",x="delta",order=orden,color="black",ax=ax)
    ax.axvline(0,ls="--",color="gray")
    ax.set(title="Estabilidad de la ganancia en los mismos cinco pliegues",xlabel="Diferencia de F1",ylabel="")
    figura(ruta,"03_estabilidad_ganancia",fig)
    fig,axs=plt.subplots(1,2,figsize=(15,6))
    pos=np.arange(len(contrastes))
    axs[0].hlines(pos,contrastes.IC_inferior,contrastes.IC_superior)
    axs[0].scatter(contrastes.delta_F1,pos)
    axs[0].axvline(0,ls="--",color="gray")
    axs[0].set(yticks=pos,yticklabels=contrastes.modelo,title="Bootstrap pareado (IC marginal 95%)",xlabel="F1 híbrido − base")
    sns.barplot(data=contrastes,y="modelo",x="p_Holm",order=contrastes.modelo,ax=axs[1])
    axs[1].axvline(.05,ls="--",color="gray")
    axs[1].set(title="Holm: diez comparaciones",xlabel="p ajustado",ylabel="",xlim=(0,1))
    figura(ruta,"04_bootstrap_holm",fig)
    fig,axs=plt.subplots(1,2,figsize=(15,6))
    sns.barplot(data=val,y="modelo",x="segundos",hue="representacion",order=orden,ax=axs[0])
    axs[0].set(xscale="log",title="Costo de entrenamiento y búsqueda",xlabel="Segundos (log)",ylabel="")
    sns.scatterplot(data=comparacion,x="factor_tiempo",y="delta_CV",s=80,ax=axs[1])
    for fila in comparacion.itertuples():
        axs[1].annotate(fila.modelo,(fila.factor_tiempo,fila.delta_CV),xytext=(4,4),textcoords="offset points",fontsize=8)
    axs[1].axhline(0,ls="--",color="gray")
    axs[1].axvline(1,ls="--",color="gray")
    axs[1].set(xscale="log",title="¿Compensa el costo adicional?",xlabel="Tiempo híbrido / base",ylabel="Ganancia F1 CV")
    axs[1].margins(x=.3,y=.15)
    figura(ruta,"05_costo_ganancia",fig)
    fig,ax=plt.subplots(figsize=(10,9))
    plot=ranking.assign(candidato=[etiqueta(f) for f in ranking.itertuples()])
    sns.barplot(data=plot,y="candidato",x="F1_CV",hue="representacion",dodge=False,ax=ax)
    ax.set(title="Ranking global: veinte candidatos",xlabel="F1 macro CV",ylabel="",xlim=(0,1))
    figura(ruta,"06_ranking_global",fig)
    # Diagnóstico de los mejores híbrido y referencia, elegidos por CV.
    clases=sorted(np.unique(yt))
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    for ax,nombre,prediccion in [(axs[0],vb.modelo.iloc[0],anteriores[vb.modelo.iloc[0]]),
                                (axs[1],vh.modelo.iloc[0]+" + grupos",pred[vh.modelo.iloc[0]])]:
        sns.heatmap(confusion_matrix(yt,prediccion,labels=clases),annot=True,fmt="d",cmap="Blues",cbar=False,
                    xticklabels=clases,yticklabels=clases,ax=ax)
        ax.set(title=nombre,xlabel="Predicción",ylabel="Real")
    figura(ruta,"07_confusiones_finalistas",fig)
