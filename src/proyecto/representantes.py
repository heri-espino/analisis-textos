"""Representantes reales de K-means y etiquetas de DBSCAN en SVD-50."""
import re
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics import adjusted_rand_score, silhouette_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_distances
from .comun import carpeta, vectorizador, guardar, SEMILLA

def ejecutar(datos):
    ruta = carpeta(9)
    original = pd.read_csv(carpeta(2) / "asignaciones.csv")
    alineado = datos[["huella"]].merge(original[["huella", "grupo"]], on="huella", validate="one_to_one")
    assert len(alineado) == len(datos) == len(original) == 3300
    v = vectorizador()
    X = v.fit_transform(datos.texto).astype(np.float32)
    Z = normalize(TruncatedSVD(50, random_state=SEMILLA).fit_transform(X))
    km = KMeans(10, n_init=10, random_state=SEMILLA).fit(Z)
    labels = alineado.grupo.to_numpy()
    assert adjusted_rand_score(labels, km.labels_) == 1
    D = cosine_distances(Z)
    np.clip(D, 0, 2, out=D)
    np.fill_diagonal(D, 0)
    filas = []
    vocab = v.get_feature_names_out()
    for g in sorted(np.unique(labels)):
        indices = np.flatnonzero(labels == g)
        # Medoide coseno = observación que minimiza la suma de distancias
        # a los miembros. Los rangos 2 y 3 son candidatos siguientes.
        costos = D[np.ix_(indices, indices)].astype(np.float64).sum(axis=1)
        centro = Z[indices].mean(axis=0)
        dist_centro = np.linalg.norm(Z[indices]-centro, axis=1)
        orden = np.lexsort((datos.huella.iloc[indices].to_numpy(), costos))[:3]
        for rango, j in enumerate(orden, 1):
            i = indices[j]
            row = X[i]
            idx = row.indices[np.argsort(row.data)[::-1][:5]]
            filas.append(dict(grupo=int(g), rango=rango, huella=datos.huella.iloc[i],
                categoria=datos.categoria.iloc[i], n_grupo=len(indices),
                distancia_coseno_media=costos[j]/(len(indices)-1),
                distancia_centroide=float(dist_centro[j]),
                terminos=", ".join(vocab[idx])))
    guardar(pd.DataFrame(filas), ruta, "representantes")
    mejores = pd.read_csv(carpeta(2) / "mejores.csv")
    fila = mejores.loc[(mejores.algoritmo=="DBSCAN") & (mejores.representacion=="SVD-50")].iloc[0]
    eps, minimo = re.fullmatch(r"eps=([^,]+), min=(\d+)", fila.parametros).groups()
    db = DBSCAN(eps=float(eps), min_samples=int(minimo), metric="precomputed", n_jobs=1).fit_predict(D)
    valido = db != -1
    sil = silhouette_score(D[np.ix_(valido,valido)], db[valido], metric="precomputed")
    ari = adjusted_rand_score(datos.categoria, db)
    nmi = normalized_mutual_info_score(datos.categoria, db)
    assert np.isclose(valido.mean(), fila.cobertura)
    assert len(np.unique(db[valido])) == fila.grupos
    assert np.isclose(sil, fila.silhouette, atol=1e-6)
    assert np.isclose(ari, fila.ARI) and np.isclose(nmi, fila.NMI)
    guardar(pd.DataFrame({"huella":datos.huella,"grupo_dbscan":db}), ruta, "dbscan")
    guardar(pd.DataFrame([dict(eps=float(eps), min_samples=int(minimo), grupos=int(fila.grupos),
        cobertura=valido.mean(), silhouette=sil, ARI=ari, NMI=nmi)]), ruta, "metricas_dbscan")
