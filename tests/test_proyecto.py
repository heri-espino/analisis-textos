import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_distances
from proyecto.comun import limpiar, vectorizador, cv
from proyecto.clasificacion import modelos, pipeline, bootstrap
from proyecto.agrupamiento import ajustar
from sklearn.cluster import KMeans

def test_limpieza():
    assert limpiar("HELLO! http://example.com name@example.com") == "hello"

def test_pipeline_sin_fuga():
    p = pipeline(LogisticRegression())
    p.fit(["planet planet orbit", "planet orbit galaxy", "hockey goal ice", "hockey goal skate"], [0, 0, 1, 1])
    assert "extraterrestre" not in p.named_steps["texto"].vocabulary_
    assert p.predict(["extraterrestre planet"]).shape == (1,)
    assert len(modelos()) == 7
    assert cv().n_splits == 5
    for modelo, _, reducir in modelos().values():
        clone(pipeline(modelo, reducir))

def test_bootstrap_pareado():
    y = np.repeat([0, 1, 2], 10)
    tabla, replicas = bootstrap(y, {"a": y, "b": y.copy(), "c": np.roll(y, 1)}, B=50)
    assert len(tabla) == 3 and replicas.shape == (50, 3)
    assert tabla.iloc[0].delta_F1 == 0 and tabla.iloc[0].p_Holm == 1
    assert (tabla.p_Holm >= tabla.p).all()

def test_agrupamiento():
    X = vectorizador().fit_transform(["planet orbit"]*4 + ["hockey ice"]*4)
    D = cosine_distances(X)
    np.fill_diagonal(D, 0)
    fila, lab = ajustar("K-means", "k=2", KMeans(2, n_init=1, random_state=42), False, X, D, "TF-IDF")
    assert fila["grupos"] == 2 and len(lab) == 8
