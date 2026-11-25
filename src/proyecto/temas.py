"""Transformación inductiva: etiquetas de grupos ajustadas sin supervisión."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, Normalizer, FunctionTransformer
from .comun import SEMILLA

class GrupoOneHot(TransformerMixin, BaseEstimator):
    """Adaptador mínimo de predict a transform; no utiliza y."""
    def __init__(self, estimador, k=10):
        self.estimador = estimador
        self.k = k

    def fit(self, X, y=None):
        self.estimador_ = clone(self.estimador).fit(X)
        self.encoder_ = OneHotEncoder(categories=[np.arange(self.k)],
                                      handle_unknown="error", sparse_output=True)
        self.encoder_.fit(self.estimador_.predict(X).reshape(-1, 1))
        return self

    def transform(self, X):
        return self.encoder_.transform(self.estimador_.predict(X).reshape(-1, 1))

def caracteristicas(reducir=False, k=10, componentes=50):
    """Conservar exactamente la rama léxica y añadir dos bloques one-hot."""
    lexico = TruncatedSVD(componentes, random_state=SEMILLA) if reducir else FunctionTransformer()
    temas = Pipeline([
        ("svd", TruncatedSVD(componentes, random_state=SEMILLA)),
        ("normalizar", Normalizer()),
        ("grupos", FeatureUnion([
            ("kmeans", GrupoOneHot(KMeans(k, n_init=10, random_state=SEMILLA), k)),
            ("gmm", GrupoOneHot(GaussianMixture(k, covariance_type="diag", n_init=2,
                                               random_state=SEMILLA), k)),
        ])),
    ])
    return FeatureUnion([("lexico", lexico), ("temas", temas)])
