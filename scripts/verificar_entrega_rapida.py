"""Ejecutar la notebook de entrega sin repositorio ni entrenamiento y empaquetar."""
from pathlib import Path
import hashlib, shutil, tempfile, time, zipfile
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
nb = nbformat.read(root/"notebook_resultados.ipynb",4)
nbformat.validate(nb)
with tempfile.TemporaryDirectory(prefix="lds1071-rapida-") as tmp:
    work = Path(tmp)
    outputs = work/"outputs"
    shutil.copytree(root/"resultados", outputs)
    def hashes():
        return {str(p.relative_to(outputs)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in outputs.rglob("*") if p.is_file()}
    antes=hashes()
    # Prohibir cualquier ajuste o descarga del corpus durante la prueba.
    guard=nbformat.v4.new_code_cell(
        "from sklearn.pipeline import Pipeline\n"
        "from sklearn.model_selection import GridSearchCV\n"
        "import sklearn.datasets\n"
        "def prohibido(*args,**kwargs): raise AssertionError('No se permite entrenar ni descargar en modo rápido')\n"
        "Pipeline.fit=prohibido\n"
        "GridSearchCV.fit=prohibido\n"
        "sklearn.datasets.fetch_20newsgroups=prohibido")
    nb.cells.insert(2, guard)
    inicio=time.perf_counter()
    NotebookClient(nb,timeout=180,kernel_name="python3",resources={"metadata":{"path":str(work)}}).execute()
    assert hashes()==antes, "La notebook modificó los resultados históricos"
    print(f"Notebook ejecutada en {time.perf_counter()-inicio:.1f} segundos sin entrenamiento")
    zip_path=root/"LDS1071_entrega_rapida.zip"
    with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED) as z:
        for nombre in ["notebook_resultados.ipynb","requirements_resultados.txt","LEEME_ENTREGA.md"]:
            z.write(root/nombre,nombre)
        for p in outputs.rglob("*"):
            if p.is_file(): z.write(p,Path("outputs")/p.relative_to(outputs))
print("Entrega lista:",zip_path)
