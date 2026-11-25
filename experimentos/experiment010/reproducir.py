"""Reproducir el corte de diez grupos y la pureza del árbol de SVD-50 sin reajustar."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree
from sklearn.metrics import normalized_mutual_info_score

def ejecutar(raiz):
    r = Path(raiz)
    L = pd.read_csv(r/"experiment008/enlace_SVD-50.csv").to_numpy()
    hojas = pd.read_csv(r/"experiment008/aglomerativo_SVD-50.csv")
    p = pd.read_csv(r/"experiment001/particiones.csv")
    assert hojas.huella.equals(p.huella)
    n = len(hojas)
    clases = sorted(p.categoria.unique())
    grupos = cut_tree(L, n_clusters=10).ravel()
    assert len(np.unique(grupos)) == 10
    counts = np.zeros((2*n-1, len(clases)), dtype=int)
    for i, c in enumerate(p.categoria):
        counts[i, clases.index(c)] = 1
    for j, (a,b,_,_) in enumerate(L):
        counts[n+j] = counts[int(a)] + counts[int(b)]
    probs = counts/counts.sum(axis=1, keepdims=True)
    pureza = np.clip((np.sum(probs**2,axis=1)-1/len(clases))/(1-1/len(clases)),0,1)
    out = r/"experiment010"
    out.mkdir(exist_ok=True)
    pd.DataFrame({"huella":hojas.huella,"grupo_aglomerativo":grupos}).to_csv(out/"aglomerativo_10.csv",index=False)
    pd.DataFrame({"nodo":np.arange(2*n-1),"n":counts.sum(axis=1),"categoria_dominante":np.array(clases)[counts.argmax(axis=1)],"pureza_gini":pureza}).to_csv(out/"pureza_nodos.csv",index=False)
    assert np.isclose(normalized_mutual_info_score(p.categoria, grupos), 0.38597509599710994)
    print(pd.Series(grupos).value_counts().sort_index())
if __name__ == "__main__":
    ejecutar(Path(__file__).resolve().parents[2]/"resultados")
