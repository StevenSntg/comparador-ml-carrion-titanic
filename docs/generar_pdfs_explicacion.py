# -*- coding: utf-8 -*-
"""Genera 2 PDFs explicativos (para exposición en clase) del código de los
modelos aplicados a la enfermedad de Carrión:

  - Carrion_RedNeuronal_explicacion.pdf
  - Carrion_ArbolDecision_explicacion.pdf

Cada PDF explica el código bloque por bloque. Usa reportlab (sin LaTeX).
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Preformatted,
                                Table, TableStyle, HRFlowable, PageBreak)


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_styles(accent):
    ss = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=ss["Title"], fontName="Helvetica-Bold",
                                 fontSize=24, leading=28, textColor=accent, spaceAfter=4),
        "subtitle": ParagraphStyle("subtitle", parent=ss["Normal"], fontName="Helvetica",
                                   fontSize=12, leading=16, textColor=colors.HexColor("#444444"),
                                   spaceAfter=2),
        "meta": ParagraphStyle("meta", parent=ss["Normal"], fontName="Helvetica-Oblique",
                               fontSize=9.5, textColor=colors.HexColor("#888888")),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold",
                             fontSize=14, leading=18, textColor=accent,
                             spaceBefore=16, spaceAfter=6),
        "body": ParagraphStyle("body", parent=ss["Normal"], fontName="Helvetica",
                               fontSize=10.5, leading=15.5, alignment=TA_JUSTIFY,
                               textColor=colors.HexColor("#1a1a1a"), spaceAfter=6),
        "note": ParagraphStyle("note", parent=ss["Normal"], fontName="Helvetica-Oblique",
                               fontSize=10, leading=14, textColor=colors.HexColor("#555555"),
                               spaceAfter=6),
        "code": ParagraphStyle("code", fontName="Courier", fontSize=8.6, leading=11.5,
                               textColor=colors.HexColor("#11243a")),
    }


def code_block(texto, st):
    # Preformatted ya escapa el markup internamente; NO volver a escapar
    # (si no, '>' se mostraría como '&gt;').
    pre = Preformatted(texto.strip("\n"), st["code"])
    tbl = Table([[pre]], colWidths=[16.4 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f3f5f8")),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#d2d9e2")),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return tbl


def make_pdf(filename, title, subtitle, accent_hex, blocks):
    accent = colors.HexColor(accent_hex)
    st = build_styles(accent)
    doc = SimpleDocTemplate(filename, pagesize=A4,
                            leftMargin=2.3 * cm, rightMargin=2.3 * cm,
                            topMargin=2.0 * cm, bottomMargin=1.8 * cm,
                            title=title, author="stevensntg")
    story = [
        Paragraph(title, st["title"]),
        Paragraph(subtitle, st["subtitle"]),
        Spacer(1, 4),
        Paragraph("Inteligencia Artificial · UNTELS · Caso: enfermedad de Carrión (MINSA) · "
                  "Comparativa Red Neuronal vs. Árbol de Decisión", st["meta"]),
        Spacer(1, 6),
        HRFlowable(width="100%", thickness=1.4, color=accent),
        Spacer(1, 10),
    ]
    for kind, content in blocks:
        if kind == "h2":
            story.append(Paragraph(content, st["h2"]))
        elif kind == "p":
            story.append(Paragraph(content, st["body"]))
        elif kind == "note":
            story.append(Paragraph(content, st["note"]))
        elif kind == "code":
            story.append(code_block(content, st))
            story.append(Spacer(1, 6))
        elif kind == "pagebreak":
            story.append(PageBreak())
    doc.build(story)
    print("OK ->", filename)


# ══════════════════════════════════════════════════════════════════════════
# PDF 1 — RED NEURONAL
# ══════════════════════════════════════════════════════════════════════════
rn_blocks = [
    ("p", "Este documento explica, <b>bloque por bloque</b>, el código del modelo de "
          "<b>Red Neuronal</b> que clasifica la fase de la enfermedad de Carrión en "
          "<b>AGUDA</b> (0) o <b>ERUPTIVA</b> (1), usando datos de vigilancia del MINSA "
          "(2000–2024, ~46 mil registros). Una red neuronal aprende ajustando los "
          "<b>pesos</b> de sus conexiones para minimizar el error de predicción."),

    ("h2", "1. Librerías y carga de datos"),
    ("code",
     'import pandas as pd, numpy as np\n'
     'training = pd.read_csv("comparador-ml/data/carrion-train.csv")'),
    ("p", "<b>pandas</b> carga el CSV de entrenamiento en una tabla (DataFrame). "
          "<b>numpy</b> se usa para las operaciones numéricas (matrices). Trabajamos con "
          "el 80% de los datos reservado para entrenar; el otro 20% (prueba) se usa al final "
          "para medir qué tan bien generaliza el modelo a casos nuevos."),

    ("h2", "2. Preprocesamiento: convertir texto a números"),
    ("p", "Una red neuronal solo entiende números, así que transformamos cada columna:"),
    ("code",
     'def edad_anios(e, t):                      # tipo_edad: A=años, M=meses, D=días\n'
     '    t = str(t).upper()\n'
     '    return e/12 if t=="M" else (e/365 if t=="D" else float(e))\n\n'
     'df["edad_anios"] = df.apply(lambda r: edad_anios(r["edad"], r["tipo_edad"]), axis=1)\n'
     'df["sexo_cod"]   = df["sexo"].apply(lambda s: 0 if str(s).upper()=="M" else 1)\n'
     'df["depto_cod"]  = df["departamento"].apply(lambda d: depto_idx[d])   # 0..18\n'
     'df["target"]     = df["enfermedad"].map({"...AGUDA":0, "...ERUPTIVA":1})'),
    ("p", "• <b>edad_anios</b>: unifica la edad a años (un bebé de 6 meses → 0.5). "
          "• <b>sexo_cod</b>: Masculino=0, Femenino=1. "
          "• <b>depto_cod</b>: cada departamento recibe un número (0 a 18). "
          "• <b>target</b>: la variable que queremos predecir, convertida a 0/1."),
    ("code",
     'columns = ["edad_anios","sexo_cod","depto_cod","ano","semana"]\n'
     'x_input = df[columns].values     # 5 columnas de entrada (features)\n'
     'y_target = df["target"].values   # la respuesta correcta (0 o 1)'),
    ("p", "<b>x_input</b> son las 5 variables que el modelo recibe; <b>y_target</b> es la "
          "respuesta correcta de cada caso, con la que el modelo compara sus predicciones "
          "durante el entrenamiento."),

    ("h2", "3. Normalización (paso clave de la red)"),
    ("code",
     'x_min = x_input.min(axis=0); x_max = x_input.max(axis=0)\n'
     'rango = np.where((x_max-x_min)==0, 1, x_max-x_min)\n'
     'x_input_n = (x_input - x_min) / rango      # cada columna queda entre 0 y 1'),
    ("p", "Las 5 variables tienen escalas muy distintas: el <b>año</b> ronda 2000, la "
          "<b>semana</b> va de 1 a 53 y la <b>edad</b> de 0 a 99. Si no se ajustan, el año "
          "(números enormes) dominaría el aprendizaje. La <b>normalización min-max</b> "
          "lleva cada columna al rango [0, 1] para que todas pesen parejo. (El Árbol de "
          "Decisión NO necesita este paso; la red sí.)"),

    ("h2", "4. Arquitectura de la red: 5 → 32 → 32 → 1"),
    ("code",
     'import keras\n'
     'from keras import layers\n'
     'model = keras.Sequential()\n'
     'model.add(layers.Dense(32, input_dim=5, activation="relu"))   # capa oculta 1\n'
     'model.add(layers.Dense(32, activation="relu", name="layer1")) # capa oculta 2\n'
     'model.add(layers.Dense(1,  activation="sigmoid", name="layer2"))  # salida'),
    ("p", "<b>Sequential</b> apila capas una tras otra. "
          "<b>input_dim=5</b>: entran las 5 variables. "
          "Dos capas ocultas de <b>32 neuronas</b> con activación <b>ReLU</b> (deja pasar "
          "lo positivo y anula lo negativo; permite aprender relaciones no lineales). "
          "La capa final tiene <b>1 neurona</b> con <b>sigmoid</b>, que devuelve una "
          "probabilidad entre 0 y 1 (probabilidad de ERUPTIVA)."),
    ("note", "Es exactamente la misma arquitectura que el ejemplo del Titanic del profesor; "
             "solo cambian los datos."),

    ("h2", "5. Compilación y entrenamiento"),
    ("code",
     'model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])\n'
     'model.fit(x_input_n, y_target, epochs=400, batch_size=256, verbose=0)'),
    ("p", "<b>loss = binary_crossentropy</b>: la función de error apropiada para clasificación "
          "de 2 clases (penaliza fuerte equivocarse con seguridad). "
          "<b>optimizer = adam</b>: el algoritmo que ajusta los pesos para reducir el error. "
          "<b>epochs = 400</b>: el modelo recorre todos los datos 400 veces. "
          "<b>batch_size = 256</b>: procesa de a 256 casos por paso (acelera el entrenamiento "
          "porque el dataset es grande; el Titanic, con 915 filas, no lo necesitaba)."),

    ("h2", "6. Evaluación del modelo"),
    ("code",
     'score = model.evaluate(x_input_n, y_target, verbose=0)\n'
     'print("Accuracy:", score[1]*100)\n\n'
     'from sklearn.metrics import confusion_matrix, classification_report\n'
     'y_sim = (model.predict(x_input_n).ravel() >= 0.5).astype(int)\n'
     'print(confusion_matrix(y_target, y_sim))\n'
     'print(classification_report(y_target, y_sim, target_names=["AGUDA","ERUPTIVA"]))'),
    ("p", "<b>accuracy</b>: porcentaje de aciertos. "
          "La <b>matriz de confusión</b> muestra aciertos y errores por clase (cuántas AGUDA "
          "predijo bien, cuántas confundió con ERUPTIVA, etc.). El <b>classification_report</b> "
          "añade precisión, recall y F1. La regla <b>&gt;= 0.5</b> convierte la probabilidad "
          "de la red en una decisión: 0.5 o más → ERUPTIVA, si no → AGUDA."),

    ("h2", "7. Predicción de un caso nuevo"),
    ("code",
     'caso = np.array([[20, 0, 1, 2004, 31]])        # edad,sexo,depto,año,semana\n'
     'caso_n = (caso - x_min) / rango                # se normaliza igual que el train\n'
     'prob = model.predict(caso_n)[0][0]             # probabilidad de ERUPTIVA'),
    ("p", "Para predecir un caso nuevo hay que normalizarlo con <b>los mismos</b> min/max del "
          "entrenamiento. Por eso esos valores se guardan en <b>norm.json</b> y se reutilizan "
          "en la página web, garantizando que el navegador calcule igual que Python."),

    ("h2", "8. Guardado y uso en la web"),
    ("code",
     'model.to_json()                 -> mimodelocarrion.json     (estructura)\n'
     'model.save_weights(...)         -> mimodelocarrion.weights.h5 (pesos)\n'
     'tensorflowjs ...                -> public/models/carrion/rn/ (formato navegador)'),
    ("p", "El modelo entrenado se guarda y se convierte a <b>TensorFlow.js</b>, que es lo que "
          "carga la página web. Son los <b>mismos pesos</b>: la web no reentrena, solo usa el "
          "modelo ya aprendido."),

    ("h2", "Resultado e interpretación"),
    ("p", "La red obtiene aproximadamente <b>72% en entrenamiento y 72% en prueba</b>. Lo "
          "importante: ambos números son casi iguales, lo que indica que la red "
          "<b>generaliza bien</b> (no memoriza). Da probabilidades prudentes en lugar de "
          "certezas absolutas — su gran ventaja frente al Árbol de Decisión, que se compara "
          "en el otro documento."),
]

# ══════════════════════════════════════════════════════════════════════════
# PDF 2 — ÁRBOL DE DECISIÓN
# ══════════════════════════════════════════════════════════════════════════
ad_blocks = [
    ("p", "Este documento explica, <b>bloque por bloque</b>, el código del modelo de "
          "<b>Árbol de Decisión</b> que clasifica la fase de la enfermedad de Carrión en "
          "<b>AGUDA</b> (0) o <b>ERUPTIVA</b> (1). Un árbol de decisión hace preguntas "
          "sucesivas sobre los datos (por ejemplo «¿edad &lt; 15?») y, según las respuestas, "
          "desciende por sus ramas hasta llegar a una hoja con la decisión final."),

    ("h2", "1. Librerías y carga de datos"),
    ("code",
     'import pandas as pd, numpy as np\n'
     'from sklearn import tree\n'
     'training = pd.read_csv("comparador-ml/data/carrion-train.csv")'),
    ("p", "Usamos <b>scikit-learn</b> (módulo <b>tree</b>), la librería estándar de Machine "
          "Learning clásico en Python. Cargamos el mismo 80% de entrenamiento que usa la red "
          "neuronal: ambos modelos comparten la <b>misma fuente de datos</b> (requisito del "
          "examen para que la comparación sea justa)."),

    ("h2", "2. Preprocesamiento (igual que la red, pero SIN normalizar)"),
    ("code",
     'df["edad_anios"] = df.apply(lambda r: edad_anios(r["edad"], r["tipo_edad"]), axis=1)\n'
     'df["sexo_cod"]   = df["sexo"].apply(lambda s: 0 if str(s).upper()=="M" else 1)\n'
     'df["depto_cod"]  = df["departamento"].apply(lambda d: depto_idx[d])\n'
     'df["target"]     = df["enfermedad"].map({"...AGUDA":0, "...ERUPTIVA":1})\n'
     'columnas = ["edad_anios","sexo_cod","depto_cod","ano","semana"]\n'
     'X = df[columnas].values\n'
     'Y = df["target"].values'),
    ("p", "El preprocesamiento es idéntico al de la red neuronal: edad a años, sexo y "
          "departamento a números, y el target a 0/1. <b>Diferencia clave</b>: el árbol "
          "<b>no necesita normalización</b>. Como toma decisiones por umbrales (por ejemplo "
          "«¿año &lt; 2005?»), la escala de cada variable no le afecta."),

    ("h2", "3. Entrenamiento del árbol (criterio de entropía)"),
    ("code",
     'miarbol = tree.DecisionTreeClassifier(criterion="entropy", random_state=42)\n'
     '# miarbol = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)  # (podado)\n'
     'miarbol = miarbol.fit(X, Y)\n'
     'print("Accuracy:", 100*miarbol.score(X, Y))'),
    ("p", "<b>criterion=\"entropy\"</b>: en cada nodo, el árbol elige la pregunta que más "
          "reduce la <b>entropía</b> (el «desorden» o mezcla de clases). Es decir, busca la "
          "división que deja los grupos lo más «puros» posible (idealmente, AGUDA por un lado "
          "y ERUPTIVA por otro). <b>fit</b> construye el árbol automáticamente a partir de los "
          "datos."),
    ("note", "La segunda línea, comentada, es la versión <b>podada</b> (max_depth=5): limitaría "
             "la profundidad para evitar el sobreajuste. Igual que en el ejemplo del profesor, "
             "se deja como referencia pero NO se usa: el árbol crece sin límite."),

    ("h2", "4. Visualización del árbol"),
    ("code",
     'import matplotlib.pyplot as plt\n'
     'from sklearn.tree import plot_tree\n'
     'plot_tree(miarbol, max_depth=3, feature_names=columnas,\n'
     '          class_names=["AGUDA","ERUPTIVA"], filled=True)\n'
     'plt.show()'),
    ("p", "<b>plot_tree</b> dibuja el árbol. Como el árbol completo es enorme (memoriza el "
          "entrenamiento), mostramos solo los <b>primeros 3 niveles</b> con <b>max_depth=3</b>. "
          "Cada caja indica la pregunta, la entropía, cuántos casos llegan y la clase "
          "predominante; <b>filled=True</b> colorea según la clase."),

    ("h2", "5. Evaluación con el conjunto de prueba"),
    ("code",
     'Xtest, Ytest = preparar(testing)\n'
     'print("Accuracy prueba:", 100*miarbol.score(Xtest, Ytest))\n\n'
     'M = confusion_matrix(Ytest, miarbol.predict(Xtest))\n'
     'sensibilidad = M[0,0] / (M[0,0] + M[1,0])\n'
     'especificidad = M[1,1] / (M[1,1] + M[0,1])'),
    ("p", "Evaluamos sobre datos que el árbol <b>no vio</b>. La <b>sensibilidad</b> mide qué "
          "tan bien detecta la clase AGUDA; la <b>especificidad</b>, qué tan bien detecta "
          "ERUPTIVA. Comparar la exactitud en entrenamiento vs. prueba revela si el modelo "
          "memorizó o realmente aprendió."),

    ("h2", "6. Guardado y uso en la web"),
    ("code",
     'import pickle\n'
     'pickle.dump(miarbol, open("miarbolcarrion.pkl","wb"))   # modelo entrenado\n\n'
     '# Exportación a JSON (nodos del árbol) para recorrerlo en el navegador:\n'
     '# children_left, children_right, feature, threshold, value  ->  ad.json'),
    ("p", "El árbol se guarda con <b>pickle</b> (formato de Python). Para la web se exporta "
          "además a un <b>JSON</b> con la estructura de nodos, y un recorrido de ~20 líneas en "
          "JavaScript reproduce exactamente las mismas decisiones. Se verificó que coincide "
          "al 100% con scikit-learn sobre las 9,224 filas de prueba."),

    ("h2", "Resultado e interpretación"),
    ("p", "El árbol obtiene cerca de <b>93% en entrenamiento pero solo 67% en prueba</b>. Esa "
          "gran diferencia es señal de <b>sobreajuste (overfitting)</b>: el árbol "
          "<b>memorizó</b> el entrenamiento (creando reglas para casos rarísimos) en lugar de "
          "aprender patrones generales. Por eso puede dar <b>100% de «certeza»</b> en casos "
          "atípicos (por ejemplo un caso en Lima, donde la enfermedad casi no existe), aunque "
          "esa certeza se base en muy pocos ejemplos."),
    ("p", "<b>Conclusión de la comparativa:</b> sobre los mismos datos, el Árbol de Decisión "
          "memoriza (alta exactitud en entrenamiento, baja en prueba) mientras que la Red "
          "Neuronal generaliza mejor (exactitud estable ~72%). El árbol gana en "
          "<b>interpretabilidad</b> (sus reglas se pueden leer y dibujar); la red gana en "
          "<b>generalización y prudencia</b> de sus probabilidades."),
]

# ══════════════════════════════════════════════════════════════════════════
# PDF 3 — CASO CLÍNICO (síntomas, dataset simulado)
# ══════════════════════════════════════════════════════════════════════════
clinico_blocks = [
    ("note", "DATASET SIMULADO con fines educativos. No proviene del MINSA ni de ninguna "
             "fuente real: los datos abiertos peruanos de vigilancia no publican síntomas por "
             "paciente. Aquí se GENERAN sintéticamente según los criterios clínicos de la "
             "literatura médica, para comparar ambos modelos sobre variables (síntomas) que sí "
             "distinguen bien cada fase."),
    ("p", "Este documento explica el código del caso <b>«Carrión por síntomas»</b>: un "
          "formulario donde se marcan síntomas y los modelos predicen la fase <b>AGUDA</b> "
          "(Fiebre de la Oroya) o <b>ERUPTIVA</b> (Verruga Peruana). Se comparan Red Neuronal "
          "y Árbol de Decisión, igual que en los otros casos."),

    ("h2", "1. Generación del dataset sintético"),
    ("code",
     'rng = np.random.RandomState(42); N = 6000\n'
     'SINTOMAS = ["fiebre_alta","palidez_anemia","fatiga_debilidad","dolor_cabeza_cuerpo",\n'
     '            "ictericia","ganglios_hinchazon","verrugas_piel","sangrado_lesiones"]\n'
     '# Probabilidad de cada síntoma SEGÚN la fase (de la clínica):\n'
     'P_AGUDA    = {"fiebre_alta":.92, "palidez_anemia":.90, ... "verrugas_piel":.05}\n'
     'P_ERUPTIVA = {"fiebre_alta":.12, "palidez_anemia":.16, ... "verrugas_piel":.90}\n'
     'for fase in fases:                      # fase sorteada 50/50\n'
     '    P = P_AGUDA if fase=="AGUDA" else P_ERUPTIVA\n'
     '    fila = {s: int(rng.rand() < P[s]) for s in SINTOMAS}   # Bernoulli(P)'),
    ("p", "Para cada paciente se sortea su fase y luego cada síntoma se activa (1) o no (0) "
          "con una probabilidad que depende de la fase. Por ejemplo, la fiebre aparece en el "
          "92% de los casos agudos pero solo en el 12% de los eruptivos; las verrugas, al "
          "revés. Así el dataset refleja la clínica real sin ser datos reales."),
    ("note", "Se añade ~8% de ruido (voltear un síntoma al azar) para simular presentaciones "
             "atípicas y evitar que la clasificación sea trivial."),

    ("h2", "2. Preparar entradas y dividir train/test"),
    ("code",
     'X = df[SINTOMAS].values.astype(float)            # 8 síntomas (0/1)\n'
     'Y = (df["fase"] == "ERUPTIVA").astype(int).values # AGUDA=0, ERUPTIVA=1\n'
     'Xtr,Xte,ytr,yte = train_test_split(X,Y, test_size=.2, stratify=Y, random_state=42)'),
    ("p", "Las 8 columnas de síntomas son la entrada; la fase es el objetivo. Reservamos 20% "
          "para prueba. Como los síntomas ya son 0/1, <b>no hace falta normalizar</b> "
          "(ni para el árbol ni para la red)."),

    ("h2", "3. Árbol de Decisión"),
    ("code",
     'from sklearn.tree import DecisionTreeClassifier\n'
     'arbol = DecisionTreeClassifier(criterion="entropy", random_state=42).fit(Xtr, ytr)\n'
     'print("train:", arbol.score(Xtr,ytr), " test:", arbol.score(Xte,yte))'),
    ("p", "El árbol aprende reglas del tipo «¿tiene verrugas? → ERUPTIVA». Con síntomas tan "
          "informativos, sus reglas son cortas y certeras."),

    ("h2", "4. Red Neuronal (8 → 32 → 32 → 1)"),
    ("code",
     'red = keras.Sequential()\n'
     'red.add(layers.Dense(32, input_dim=8, activation="relu"))\n'
     'red.add(layers.Dense(32, activation="relu"))\n'
     'red.add(layers.Dense(1, activation="sigmoid"))\n'
     'red.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])\n'
     'red.fit(Xtr, ytr, epochs=200, batch_size=64, verbose=0)'),
    ("p", "Misma arquitectura que los otros casos, pero con <b>8 entradas</b> (una por "
          "síntoma). La capa final sigmoide da la probabilidad de ERUPTIVA."),

    ("h2", "5. Predicción de un caso por síntomas"),
    ("code",
     '# [fiebre,palidez,fatiga,dolor,ictericia,ganglios,verrugas,sangrado]\n'
     'caso_sistemico = np.array([[1,1,1,1,0,0,0,0]])   # -> AGUDA\n'
     'caso_cutaneo   = np.array([[0,0,0,0,0,0,1,1]])   # -> ERUPTIVA\n'
     'arbol.predict(caso_cutaneo)            # 1 (ERUPTIVA)\n'
     'red.predict(caso_cutaneo)[0][0] >= 0.5 # True (ERUPTIVA)'),
    ("p", "Un paciente con síntomas sistémicos (fiebre, anemia, fatiga) es clasificado como "
          "AGUDA por ambos modelos; uno con verrugas que sangran, como ERUPTIVA. Es el mismo "
          "razonamiento que hace el formulario web."),

    ("h2", "Resultado e interpretación"),
    ("p", "Aquí <b>ambos modelos alcanzan ~98% en prueba</b> y coinciden casi siempre. Esto "
          "contrasta con el caso basado en datos del MINSA (edad, sexo, zona, fecha), donde el "
          "árbol sobreajustaba y la red generalizaba mejor."),
    ("p", "<b>Lección de la comparativa:</b> la <b>calidad de las variables importa tanto o "
          "más que el modelo</b>. Con predictores realmente informativos (síntomas), Red "
          "Neuronal y Árbol rinden parecido; las diferencias entre modelos se notan sobre todo "
          "cuando los datos son débiles o ruidosos. (Recordatorio: dataset simulado, no apto "
          "para diagnóstico real.)"),
]

make_pdf("Carrion_RedNeuronal_explicacion.pdf",
         "Red Neuronal aplicada a la Enfermedad de Carrión",
         "Explicación del código, paso a paso (material de exposición)",
         "#b4561f", rn_blocks)

make_pdf("Carrion_ArbolDecision_explicacion.pdf",
         "Árbol de Decisión aplicado a la Enfermedad de Carrión",
         "Explicación del código, paso a paso (material de exposición)",
         "#1f7a4d", ad_blocks)

make_pdf("Carrion_Clinico_explicacion.pdf",
         "Carrión por síntomas — Red Neuronal vs. Árbol (dataset simulado)",
         "Explicación del código, paso a paso (material de exposición)",
         "#157a8a", clinico_blocks)
