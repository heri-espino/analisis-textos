"""Validación de la notebook autocontenida en un directorio vacío."""
from pathlib import Path
import tempfile, sys, os
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
nb = nbformat.read(root / "notebook_autonoma.ipynb", 4)
nbformat.validate(nb)
for celda in nb.cells:
    if celda.cell_type == "code":
        compile(celda.source, "celda", "exec")
        assert "%%writefile" not in celda.source
        assert "from proyecto import" not in celda.source
        assert "pd.read_csv" not in celda.source
        assert "joblib.load" not in celda.source
        assert "LDS1071" not in celda.source
if "--sintaxis" in sys.argv:
    print("Formato, sintaxis y ausencia de dependencias locales verificados.")
    sys.exit(0)

completa = "--completa" in sys.argv
with tempfile.TemporaryDirectory(prefix="notebook-") as tmp:
    if not completa:
        tags = {"configuracion", "definicion", "carga", "eda"}
        c = [x for x in nb.cells if x.cell_type=="code" and tags.intersection(x.metadata.get("tags", []))]
        c.append(nbformat.v4.new_code_cell(
            "m = pipeline(LogisticRegression(max_iter=1000, random_state=42))\n"
            "t = datos[datos.particion.eq('train')]\n"
            "m.fit(t.texto,t.categoria)\n"
            "assert len(m.predict(t.texto.iloc[:12]))==12\n"
            "assert len(leer('eda','particiones'))==3300\n"
            "assert len(FIGURAS)>0\n"
            "print('Carga pública, EDA en memoria y pipeline verificados.')"))
        nb = nbformat.v4.new_notebook(cells=c, metadata=nb.metadata)
    client = NotebookClient(nb, timeout=None, kernel_name="python3",
                            resources={"metadata":{"path":tmp}})
    client.execute()
    if completa:
        nbformat.write(nb, root / "notebook_autonoma_ejecutada.ipynb")
    else:
        for output in nb.cells[-1].outputs:
            if output.output_type=="stream":
                print(output.text)
print("Notebook validada en directorio independiente.")
