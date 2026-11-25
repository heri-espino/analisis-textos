"""Crear la notebook y el reporte de cinco capítulos a partir de resultados guardados."""
from pathlib import Path
from xml.sax.saxutils import escape
import json
import nbformat as nbf
from nbclient import NotebookClient
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from .comun import RAIZ, RESULTADOS

def leer(n, nombre):
    return pd.read_csv(RESULTADOS / f"experiment{n:03d}" / f"{nombre}.csv")

def capitulos():
    """Las interpretaciones numéricas se calculan a partir de las tablas reales."""
    e = leer(1, "eda")
    rep = leer(1, "representaciones")
    var = leer(1, "varianza_svd")
    cl = leer(2, "configuraciones")
    sel = cl.loc[cl.seleccionado].iloc[0]
    mejores = leer(2, "mejores")
    externo = cl.loc[cl.ARI.idxmax()]
    val = leer(3, "validacion")
    prueba = leer(3, "metricas_prueba")
    mejor = val.iloc[0]
    individual = val.loc[val.familia.eq("Individual")].iloc[0]
    ensamble = val.loc[val.familia.eq("Ensamble")].iloc[0]
    rapido = val.loc[val.segundos.idxmin()]
    test = prueba.loc[prueba.modelo.eq(mejor.modelo)].iloc[0]
    pares = leer(3, "errores_por_par")
    error = pares.iloc[0]
    holm = leer(3, "bootstrap_holm")
    r = leer(4, "representacion")
    d = leer(4, "dimensionalidad")
    s = leer(4, "componentes_svd")
    c = leer(4, "complejidad")
    rbest = r.loc[r.mean_test_score.idxmax()]
    dbest = d.loc[d.mean_test_score.idxmax()]
    sbest = s.loc[s.mean_test_score.idxmax()]
    cbest = c.loc[c.mean_test_score.idxmax()]
    brecha = c.error_validacion - c.error_entrenamiento
    bmax = c.loc[brecha.idxmax()]
    train = int(e.documentos.sum())
    total = len(leer(1, "particiones"))
    cv_gap = test.F1 - mejor.F1_CV
    tablas = [
        ("1. Preparación y representación de textos", [
            f"Objetivo: descubrir temas sin etiquetas y clasificar documentos nuevos. Se emplea 20 Newsgroups: {total} documentos en seis categorías, {train} de entrenamiento y {total-train} de prueba. Se respeta la partición original y se selecciona un número fijo por categoría. Se eliminan cabeceras, firmas, citas, URL, correos y duplicados exactos. La tokenización conserva palabras inglesas y se excluyen palabras funcionales al vectorizar.",
            "BoW representa frecuencias absolutas; TF-IDF pondera los términos por su frecuencia inversa documental suavizada y aplica normalización L2. El vocabulario se aprende únicamente del entrenamiento para el EDA supervisado. Los términos con frecuencia documental menor que dos se excluyen.",
            f"Las medianas de longitud por clase van de {e.mediana_palabras.min():.0f} a {e.mediana_palabras.max():.0f} palabras. Las matrices tienen {rep.caracteristicas.min():,}–{rep.caracteristicas.max():,} características y {100*rep.esparsidad.min():.2f}–{100*rep.esparsidad.max():.2f}% de ceros. SVD con 100 componentes conserva {100*var.varianza_acumulada.iloc[-1]:.2f}% de la varianza según TruncatedSVD; al no centrarse la matriz, no equivale exactamente a PCA. La elevada esparsidad favorece algoritmos lineales y almacenamiento disperso.",
        ], [(1,"eda",None),(1,"representaciones",None)], 1),
        ("2. Aprendizaje no supervisado", [
            "Se ajusta de forma independiente una representación TF-IDF en los mismos documentos del corpus completo, sin entregar las etiquetas a los estimadores. K-means minimiza dispersión euclidiana; el aglomerativo utiliza enlace promedio y distancia coseno; DBSCAN busca regiones densas y puede declarar ruido; la mezcla gaussiana usa EM, covarianzas diagonales y el espacio SVD-50. Se compara TF-IDF original y SVD normalizado cuando es viable. La proyección en dos componentes solo ilustra la geometría.",
            "Se experimenta con k=3, 4, 6, 8 y 10 y, para DBSCAN, cinco radios y dos umbrales de vecinos. Silhouette coseno mide separación interna. Se excluye el ruido de su cálculo, se informa cobertura y se selecciona por cobertura × (1 + silhouette)/2 para penalizar el descarte. ARI y NMI se calculan únicamente después de congelar la selección; no se selecciona por las etiquetas.",
            f"La configuración seleccionada internamente es {sel.algoritmo}, {sel.parametros}, {sel.representacion}: {int(sel.grupos)} grupos, cobertura {sel.cobertura:.1%}, silhouette {sel.silhouette:.4f}, ARI {sel.ARI:.4f} y NMI {sel.NMI:.4f}. En contraste, el mayor ARI retrospectivo es {externo.ARI:.4f} ({externo.algoritmo}, {externo.representacion}, {externo.parametros}). Un ARI cercano a cero señala concordancia ajustada por azar escasa; uno indica equivalencia perfecta. La geometría y las etiquetas temáticas no tienen por qué coincidir.",
            "Las tablas de mejores configuraciones muestran el efecto de reducir la dimensión por algoritmo. No se compara GMM en el espacio original: una mezcla de alta dimensión resultaría más costosa y los vectores esparsos no satisfacen bien una hipótesis gaussiana. Se interpretan los grupos con términos de mayor TF-IDF promedio y extractos de dos documentos representativos. Los grupos de ruido, si aparecen, no representan necesariamente un tema.",
        ], [(2,"mejores",["algoritmo","representacion","parametros","grupos","cobertura","silhouette","ARI","NMI","segundos_ajuste"]),
            (2,"interpretacion",["grupo","documentos","palabras","ejemplo_1"])], 2),
        ("3. Aprendizaje supervisado", [
            "Se comparan regresión logística, SGD logístico, árbol de decisión, Random Forest, AdaBoost, Gradient Boosting y SVM con kernel RBF, además de tres ensambles. Las transformaciones se ajustan dentro de Pipeline en cada pliegue. Gradient Boosting usa SVD-50 para limitar su costo. GridSearchCV selecciona por F1 macro en cinco pliegues estratificados comunes, sin acceso a prueba. F1 macro promedia la media armónica de precisión y exhaustividad por clase; las demás métricas se calculan también con promedio macro salvo Accuracy.",
            "Bagging entrena veinte regresiones logísticas sobre remuestras bootstrap de los vectores TF-IDF y promedia probabilidades. La votación dura combina las siete familias con configuraciones preespecificadas. El stacking usa estas mismas siete familias, probabilidades o márgenes fuera de pliegue y XGBoost como metaclasificador (profundidades 2/3, 50/100 árboles). Cada fold externo contiene su propio stacking interno de cinco pliegues; la búsqueda del metaestimador también se limita al entrenamiento externo. La prueba permanece reservada. Las bases de los ensambles se fijan a priori para no reutilizar una selección que vio la validación externa.",
            f"El mejor individual es {individual.modelo} (F1 CV={individual.F1_CV:.4f}); el mejor ensamble es {ensamble.modelo} (F1 CV={ensamble.F1_CV:.4f}); la diferencia es {ensamble.F1_CV-individual.F1_CV:+.4f}. El menor tiempo total de búsqueda y reajuste corresponde a {rapido.modelo} ({rapido.segundos:.2f} s). Los tiempos son de pared y dependen del hardware y del paralelismo; incluyen diferentes cargas de búsqueda. Se registran por separado ajuste final y predicción.",
            f"El modelo seleccionado por CV es {mejor.modelo}, con F1 macro {mejor.F1_CV:.4f} (DE entre pliegues {mejor.DE_CV:.4f}); obtiene Accuracy {test.Accuracy:.4f}, Precision {test.Precision:.4f}, Recall {test.Recall:.4f} y F1 macro {test.F1:.4f} en prueba. Su diferencia prueba − CV es {cv_gap:+.4f}. La mayor confusión fuera de la diagonal es {error.real} → {error.predicha} ({int(error.errores)} casos); revisar extractos permite distinguir solapamiento léxico, ambigüedad y documentos cortos.",
            f"Los tres primeros de CV ({', '.join(val.modelo.head(3))}) se comparan con 2,000 réplicas bootstrap pareadas y estratificadas por etiqueta real. Se calculan diferencias de F1, intervalos percentiles marginales al 95% y p bilaterales aproximados mediante la distribución centrada bajo H0. Holm controla la familia de tres contrastes al 5%; se rechazan {int(holm.rechaza_H0.sum())} de tres. Los IC marginales no son intervalos simultáneos ajustados; no rechazar no demuestra equivalencia. La inferencia está condicionada a los modelos entrenados y al muestreo de prueba.",
        ], [(3,"validacion",["modelo","F1_CV","DE_CV","segundos","segundos_refit"]),(3,"metricas_prueba",None),(3,"mejora_ensambles",None),
            (3,"errores_por_par",["real","predicha","errores"]),
            (3,"bootstrap_holm",["modelo_A","modelo_B","delta_F1","IC_inferior","IC_superior","p_Holm"])], 3),
        ("4. Generalización y complejidad", [
            "Se usa una regresión logística de referencia para aislar cada factor, manteniendo fijos datos, pliegues, limpieza y los restantes hiperparámetros. Se ejecutan estos estudios antes de consultar prueba. Las puntuaciones son de validación cruzada, no de prueba; las estimaciones de la mejor configuración tienen sesgo de selección.",
            f"Representación: {rbest.representacion} logra el mayor F1 medio de CV ({rbest.mean_test_score:.4f}) con C=1. TF-IDF reduce el peso de términos frecuentes y la influencia de la longitud; BoW conserva el conteo absoluto. La diferencia observada puede depender del clasificador y su regularización.",
            f"Dimensionalidad: el mejor máximo de vocabulario ensayado es {int(dbest.caracteristicas_max):,} (F1={dbest.mean_test_score:.4f}); es un límite, no necesariamente el número efectivo de columnas. En el estudio SVD, {int(sbest.componentes)} componentes logran el mayor F1 ({sbest.mean_test_score:.4f}). Comprimir puede eliminar ruido, pero también vocabulario discriminativo. Comparar con TF-IDF sin reducir permite evaluar el costo de la compresión.",
            f"Complejidad: C es el inverso de la fuerza de regularización. El mejor C exploratorio es {cbest.C:g} (F1={cbest.mean_test_score:.4f}); la mayor brecha validación − entrenamiento en error 1−F1 es {brecha.max():.4f} (C={bmax.C:g}). Una brecha positiva puede ser compatible con sobreajuste; una brecha pequeña con error alto puede ser compatible con subajuste. Esta curva no usa prueba.",
        ], [(4,"representacion",["representacion","mean_train_score","mean_test_score","std_test_score"]),
            (4,"dimensionalidad",["caracteristicas_max","mean_train_score","mean_test_score"]),
            (4,"componentes_svd",["componentes","mean_train_score","mean_test_score"]),
            (4,"complejidad",["C","error_entrenamiento","error_validacion"])], 4),
        ("5. Discusión y conclusiones", [
            f"La clasificación supervisada y el agrupamiento resuelven problemas distintos: el primero aprende etiquetas y obtiene F1 de prueba {test.F1:.4f} con {mejor.modelo}; el segundo recupera estructura no etiquetada y obtiene ARI {sel.ARI:.4f} para la configuración elegida internamente. Las escalas y objetivos de ambas métricas son diferentes, por lo que no se deben restar ni comparar como si midieran lo mismo.",
            f"En los estudios controlados, la mejor representación fue {rbest.representacion}, el mejor máximo de vocabulario fue {int(dbest.caracteristicas_max):,}, la mejor dimensión latente ensayada fue {int(sbest.componentes)} y el mejor C exploratorio fue {cbest.C:g}. Son conclusiones locales a los rangos y al modelo de referencia, no óptimos universales. El modelo final se escogió por CV, no por su posición en prueba.",
            "Limitaciones: corpus histórico en inglés, seis temas seleccionados y balance artificial; eliminación de duplicados exactos sin garantía de independencia entre hilos relacionados; una sola partición oficial, búsqueda de hiperparámetros limitada, posibles documentos casi vacíos tras eliminar palabras funcionales, y representación léxica sin orden ni semántica contextual. El bootstrap no refleja la variabilidad del entrenamiento ni la deriva temporal. Una extensión sería evaluar todos los temas, particiones agrupadas por hilo, n-gramas, embeddings y validación en otro corpus; no debe reajustarse el modelo con la prueba ya evaluada.",
            "Fuente de datos y metodología de descarga: scikit-learn, The 20 newsgroups text dataset, https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset. Las tablas completas, configuraciones, predicciones, réplicas y código se conservan en el repositorio.",
        ], [], None),
    ]
    return tablas

def seleccion_tabla(n, nombre, columnas):
    tabla = leer(n, nombre)
    if columnas:
        tabla = tabla[columnas]
    if nombre == "errores_por_par":
        tabla = tabla.head(5)
    return tabla

def imagenes(n):
    return sorted((RESULTADOS / f"experiment{n:03d}").glob("*.png")) if n else []

def notebook(caps):
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python (analisis-textos)", "language": "python", "name": "python3"}
    nb.cells = [nbf.v4.new_markdown_cell("# Análisis y clasificación de textos\n\nProyecto final · 20 Newsgroups\n\nEsta notebook integra los resultados guardados. Para reproducir desde cero, ejecutar python -m proyecto en el entorno Conda. No reajusta modelos ni usa prueba para elegir hiperparámetros."),
                nbf.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\nimport seaborn as sns\nsns.set_theme(style='whitegrid', palette='dark')\nRESULTADOS = Path('resultados')")]
    for titulo, parrafos, tablas, n in caps:
        nb.cells.append(nbf.v4.new_markdown_cell("## " + titulo + "\n\n" + "\n\n".join(parrafos)))
        for num, nombre, columnas in tablas:
            codigo = f"tabla = pd.read_csv(RESULTADOS / 'experiment{num:03d}' / '{nombre}.csv')\n"
            if columnas:
                codigo += f"tabla = tabla[{columnas!r}]\n"
            if nombre == "errores_por_par":
                codigo += "tabla = tabla.head(5)\n"
            nb.cells.append(nbf.v4.new_markdown_cell("### Tabla: " + nombre.replace("_", " ")))
            nb.cells.append(nbf.v4.new_code_cell(codigo + "display(tabla.round(4))"))
        for img in imagenes(n):
            nb.cells.append(nbf.v4.new_code_cell(f"display(Image(filename={str(img.relative_to(RAIZ).as_posix())!r}))"))
    NotebookClient(nb, timeout=120, kernel_name="python3", resources={"metadata":{"path":str(RAIZ)}}).execute()
    nbf.write(nb, RAIZ / "notebook_final.ipynb")

def pdf(caps):
    estilos = getSampleStyleSheet()
    estilos["BodyText"].fontSize = 10
    estilos["BodyText"].leading = 14
    estilos["BodyText"].spaceAfter = 8
    estilos["Normal"].fontSize = 8
    estilos["Normal"].leading = 10
    historia = [Paragraph("Análisis y clasificación de textos", estilos["Title"]),
                Paragraph("Proyecto final · 20 Newsgroups · Informe técnico", estilos["Heading2"])]
    ancho = A4[0]-84
    for i, (titulo, parrafos, tablas, n) in enumerate(caps):
        if i:
            historia.append(PageBreak())
        historia.append(Paragraph(escape(titulo), estilos["Heading1"]))
        for p in parrafos:
            historia.append(Paragraph(escape(p), estilos["BodyText"]))
        for num, nombre, columnas in tablas:
            df = seleccion_tabla(num, nombre, columnas)
            historia.append(Paragraph(escape(nombre.replace("_", " ").capitalize()), estilos["Heading3"]))
            if nombre == "interpretacion":
                for _, fila in df.iterrows():
                    texto = f"Grupo {fila.grupo} ({fila.documentos} documentos): {fila.palabras}. Ejemplo: {fila.ejemplo_1}."
                    historia.append(Paragraph(escape(texto), estilos["BodyText"]))
                continue
            filas = [[Paragraph(escape(str(c)), estilos["Normal"]) for c in df.columns]]
            for fila in df.itertuples(index=False, name=None):
                filas.append([Paragraph(escape(f"{x:.4f}" if isinstance(x, float) else str(x)), estilos["Normal"]) for x in fila])
            tabla = Table(filas, colWidths=[ancho/len(df.columns)]*len(df.columns), repeatRows=1)
            tabla.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#DDE6ED")),
                                       ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F2F5F7")]),
                                       ("VALIGN",(0,0),(-1,-1),"TOP"),
                                       ("LINEBELOW",(0,0),(-1,0),.5,colors.HexColor("#607D8B"))]))
            historia.extend([tabla, Spacer(1, 9)])
        for img in imagenes(n):
            w,h = ImageReader(str(img)).getSize()
            escala = min(ancho/w, 350/h)
            historia.append(Paragraph(escape(img.stem[3:].replace("_"," ").capitalize()), estilos["Heading3"]))
            historia.append(Image(str(img), width=w*escala, height=h*escala))
            historia.append(Spacer(1, 8))
    def pie(canvas, doc):
        canvas.setFont("Helvetica", 9)
        canvas.drawString(42, 27, "Análisis y clasificación de textos")
        canvas.drawRightString(A4[0]-42, 27, f"Página {doc.page}")
    SimpleDocTemplate(str(RAIZ / "reporte.pdf"), pagesize=A4, rightMargin=42, leftMargin=42,
                      topMargin=42, bottomMargin=42, title="Análisis y clasificación de textos").build(
        historia, onFirstPage=pie, onLaterPages=pie)

def ejecutar():
    caps = capitulos()
    notebook(caps)
    pdf(caps)
