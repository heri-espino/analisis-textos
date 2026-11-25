from pathlib import Path
import json
import re
import hashlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

SEMILLA = 42
RAIZ = Path(__file__).resolve().parents[2]
RESULTADOS = RAIZ / "resultados"
CATEGORIAS = {
    "comp.graphics": "Gráficos", "rec.autos": "Automóviles",
    "rec.sport.hockey": "Hockey", "sci.med": "Medicina",
    "sci.space": "Espacio", "talk.politics.mideast": "Oriente Medio",
}
sns.set_theme(style="whitegrid", palette="dark")

def limpiar(texto):
    """Quitar URL, correos y espacios; conservar letras y contexto inglés."""
    texto = re.sub(r"https?://\S+|www\.\S+|\b\S+@\S+\b", " ", texto.lower())
    return " ".join(re.findall(r"[a-z]+(?:'[a-z]+)?", texto))

def cargar():
    """Muestreo por categoría en las particiones originales, sin duplicados."""
    filas, vistos = [], set()
    for parte, cantidad in [("train", 400), ("test", 150)]:
        datos = fetch_20newsgroups(subset=parte, categories=list(CATEGORIAS),
                                  remove=("headers", "footers", "quotes"),
                                  random_state=SEMILLA, data_home=RAIZ / "data")
        grupos = {c: [] for c in CATEGORIAS}
        for texto, etiqueta in zip(datos.data, datos.target):
            texto = limpiar(texto)
            huella = hashlib.sha256(texto.encode()).hexdigest()
            if len(texto.split()) < 5 or huella in vistos:
                continue
            vistos.add(huella)
            categoria = datos.target_names[etiqueta]
            grupos[categoria].append((texto, huella))
        for categoria, documentos in grupos.items():
            if len(documentos) < cantidad:
                raise ValueError(f"Documentos insuficientes: {categoria}, {parte}: {len(documentos)}")
            rng = np.random.default_rng(SEMILLA)
            for i in rng.choice(len(documentos), cantidad, replace=False):
                texto, huella = documentos[i]
                filas.append((texto, CATEGORIAS[categoria], parte, huella))
    return pd.DataFrame(filas, columns=["texto", "categoria", "particion", "huella"]).sample(
        frac=1, random_state=SEMILLA).reset_index(drop=True)

def carpeta(numero):
    ruta = RESULTADOS / f"experiment{numero:03d}"
    ruta.mkdir(parents=True, exist_ok=True)
    return ruta

def guardar(tabla, ruta, nombre):
    tabla.to_csv(ruta / f"{nombre}.csv", index=False)

def figura(ruta, nombre, fig=None):
    fig = fig or plt.gcf()
    fig.tight_layout()
    fig.savefig(ruta / f"{nombre}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

def vectorizador(tipo="TF-IDF", max_features=10000):
    clase = TfidfVectorizer if tipo == "TF-IDF" else CountVectorizer
    return clase(stop_words="english", min_df=2, max_features=max_features,
                 token_pattern=r"(?u)\b[a-z]{2,}\b")

def cv():
    return StratifiedKFold(5, shuffle=True, random_state=SEMILLA)

def entrenamiento(datos):
    return datos.loc[datos.particion.eq("train")]

def json_guardar(objeto, ruta):
    ruta.write_text(json.dumps(objeto, ensure_ascii=False, indent=2), encoding="utf-8")
