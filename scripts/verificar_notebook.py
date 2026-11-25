"""Verificar la notebook sin importar el paquete instalado del repositorio."""
from pathlib import Path
import tempfile, shutil, sys
import nbformat
from nbclient import NotebookClient

root=Path(__file__).resolve().parents[1]
nb=nbformat.read(root/"notebook_autonoma.ipynb",4)
nbformat.validate(nb)
completa="--completa" in sys.argv
with tempfile.TemporaryDirectory(prefix="lds1071-notebook-") as tmp:
    work=Path(tmp)
    # Reusar únicamente la caché del corpus público, si existe.
    if (root/"data").is_dir():
        shutil.copytree(root/"data", work/"data")
    if not completa:
        # Las celdas de escritura se ejecutan realmente; luego EDA y prueba de importación.
        c=[x for x in nb.cells if x.cell_type=="code"]
        setup=c[0]
        modulos=[x for x in c if x.source.startswith("%%writefile")]
        eda=next(x for x in c if "datos = comun.cargar()" in x.source)
        test=nbformat.v4.new_code_cell(
            "from proyecto import agrupamiento, proyecciones, clasificacion, hibrido, representantes, diagnostico_ampliado, generalizacion\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "m=clasificacion.pipeline(LogisticRegression(max_iter=1000, random_state=42))\n"
            "t=datos[datos.particion.eq('train')]\n"
            "m.fit(t.texto,t.categoria)\n"
            "assert len(m.predict(t.texto.iloc[:12]))==12\n"
            "assert comun.RAIZ == Path.cwd()\n"
            "print('Carga, EDA, módulos y pipeline supervisado verificados.')")
        nb=nbformat.v4.new_notebook(cells=[setup,*modulos,eda,test],metadata=nb.metadata)
    client=NotebookClient(nb,timeout=None,kernel_name="python3",resources={"metadata":{"path":str(work)}})
    client.execute()
    if completa:
        nbformat.write(nb,root/"notebook_autonoma_ejecutada.ipynb")
    else:
        for output in nb.cells[-1].outputs:
            if output.output_type=="stream": print(output.text)
print("Notebook validada en directorio temporal independiente.")
