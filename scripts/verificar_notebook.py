"""Verificar la notebook sin importar el paquete instalado del repositorio."""
from pathlib import Path
import tempfile, shutil, sys, os
import nbformat
from nbclient import NotebookClient

root=Path(__file__).resolve().parents[1]
nb=nbformat.read(root/"notebook_autonoma.ipynb",4)
nbformat.validate(nb)
completa="--completa" in sys.argv
# Validar el código normal y el texto ensamblado de cada módulo.
for celda in nb.cells:
    if celda.cell_type=="code" and not celda.source.startswith("%%writefile"):
        compile(celda.source, "celda", "exec")
if "--sintaxis" in sys.argv:
    print("Formato y código Python validados; los módulos se verifican al ensamblar.")
    sys.exit(0)
with tempfile.TemporaryDirectory(prefix="lds1071-notebook-") as tmp:
    work=Path(tmp)
    # Reusar únicamente la caché del corpus público, si existe.
    if not completa and (root/"data").is_dir():
        shutil.copytree(root/"data", work/"data")
    if not completa:
        # Las celdas de escritura se ejecutan realmente; luego EDA y prueba de importación.
        c=[x for x in nb.cells if x.cell_type=="code"]
        setup=c[:2]
        modulos=[x for x in c if x.source.startswith("%%writefile") or x.source.startswith("ensamblar(")]
        auxiliares=[x for x in c if x.source.startswith("import pandas as pd") or x.source.startswith("def ejecutar(modulo)") or x.source.startswith("datos = comun.cargar()") or x.source.startswith("ejecutar(preparacion)")]
        test=nbformat.v4.new_code_cell(
            "from proyecto import agrupamiento, proyecciones, clasificacion, hibrido, representantes, diagnostico_ampliado, generalizacion\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "m=clasificacion.pipeline(LogisticRegression(max_iter=1000, random_state=42))\n"
            "t=datos[datos.particion.eq('train')]\n"
            "m.fit(t.texto,t.categoria)\n"
            "assert len(m.predict(t.texto.iloc[:12]))==12\n"
            "assert comun.RAIZ == Path.cwd()\n"
            "print('Carga, EDA, módulos y pipeline supervisado verificados.')")
        nb=nbformat.v4.new_notebook(cells=[*setup,*modulos,*auxiliares,test],metadata=nb.metadata)
    os.environ["LDS1071_VALIDACION"]="1"
    client=NotebookClient(nb,timeout=None,kernel_name="python3",resources={"metadata":{"path":str(work)}})
    client.execute()
    if completa:
        nbformat.write(nb,root/"notebook_autonoma_ejecutada.ipynb")
    else:
        for output in nb.cells[-1].outputs:
            if output.output_type=="stream": print(output.text)
print("Notebook validada en directorio temporal independiente.")
