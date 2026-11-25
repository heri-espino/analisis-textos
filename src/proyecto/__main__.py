"""Punto de entrada reproducible."""
import argparse
import joblib
from threadpoolctl import threadpool_limits
from . import preparacion, agrupamiento, clasificacion, generalizacion
from .comun import cargar, carpeta, limpiar

def main():
    parser = argparse.ArgumentParser(description="Proyecto de análisis de textos")
    parser.add_argument("etapa", nargs="?", choices=["experiment001", "experiment002", "experiment003", "experiment004", "experiment005", "entrega"])
    parser.add_argument("--texto", help="Clasificar un documento nuevo")
    args = parser.parse_args()
    if args.texto is not None:
        modelo = joblib.load(carpeta(3) / "modelo.joblib")
        print(modelo.predict([limpiar(args.texto)])[0])
        return
    etapas = {"experiment001": preparacion, "experiment002": agrupamiento,
              "experiment004": generalizacion, "experiment003": clasificacion}
    if args.etapa != "entrega":
        datos = cargar()
        with threadpool_limits(limits=1):
            for nombre, modulo in etapas.items():
                if args.etapa in (None, nombre) or (args.etapa == "experiment005" and nombre == "experiment003"):
                    print(f"Ejecutando {nombre}", flush=True)
                    modulo.ejecutar(datos)
    if args.etapa in (None, "entrega"):
        from .entrega import ejecutar
        ejecutar()

if __name__ == "__main__":
    main()
