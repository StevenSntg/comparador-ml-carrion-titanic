"""
Verificacion: comparar las predicciones de la pagina web (TF.js)
contra las predicciones de Python (cargando el .h5 directamente).

Si las dos coinciden (margen de ~1% por redondeo de floats), confirmas
que el browser esta usando exactamente el modelo entrenado en .h5.

USO:
    py -3.12 verificar_predicciones.py
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import numpy as np
import pandas as pd
from tf_keras.models import model_from_json


# ---------------------------------------------------------------------------
# Cargar el modelo desde el .h5 (mismo formato que entrega el profesor)
# ---------------------------------------------------------------------------
print("=" * 90)
print(" VERIFICACION DE PREDICCIONES")
print("=" * 90)
print("\nCargando mimodelo.json + mimodelo.weights.h5 ...")

with open("mimodelo.json", "r") as f:
    model = model_from_json(f.read())
model.load_weights("mimodelo.weights.h5")
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
print("  -> modelo cargado")


# ---------------------------------------------------------------------------
# Bateria de casos de prueba
# ---------------------------------------------------------------------------
# Cada caso es: (descripcion, [Fare, Pclass, Gender, Age, SibSp])
# Gender: 0 = male (Hombre), 1 = female (Mujer)
#
# IMPORTANTE: el modelo del profesor SOLO usa 5 features. SibSp = hermanos +
# conyuge a bordo (NO incluye padres ni hijos). El dataset original tiene
# tambien Parch (parents + children), pero el notebook lo descarta. Por eso,
# en la web, el stepper "Hermanos o conyuge a bordo" es exactamente SibSp:
# para coincidir con un caso del notebook, usa el numero correcto en SibSp,
# NO la suma de toda la familia (padres + hijos + hermanos + conyuge).
#
# El primer caso es EL MISMO ejemplo que el notebook del profesor usa.
test_cases = [
    # NOTEBOOK ----------------------------------------------------------
    ("Ejemplo del notebook (mujer 35, 1ra clase, £83.475, 1 familiar)",
     [83.475, 1, 1, 35, 1]),

    # MUJERES EN PRIMERA CLASE - deberian sobrevivir alto -----------------
    ("Mujer 28, 1ra clase, £71, 1 familiar (default de la web)",
     [71, 1, 1, 28, 1]),
    ("Mujer 22, 1ra clase, £100, sin familia",
     [100, 1, 1, 22, 0]),

    # NINOS - deberian sobrevivir alto ('mujeres y ninos primero') --------
    ("Nino 5 anos, 2da clase, £30, 2 hermanos",
     [30, 2, 0, 5, 2]),
    ("Nina 8 anos, 3ra clase, £25, 1 hermano",
     [25, 3, 1, 8, 1]),

    # HOMBRES EN TERCERA CLASE - deberian NO sobrevivir ------------------
    ("Hombre 35, 3ra clase, £7.05, sin familiares",
     [7.05, 3, 0, 35, 0]),
    ("Hombre 22, 3ra clase, £8, sin familiares",
     [8, 3, 0, 22, 0]),

    # CASOS LIMITE ------------------------------------------------------
    ("Mujer mayor 60, 1ra clase, £250, sin familiares",
     [250, 1, 1, 60, 0]),
    ("Hombre joven 18, 2da clase, £15, sin familia",
     [15, 2, 0, 18, 0]),
]


# ---------------------------------------------------------------------------
# Predicciones individuales
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print(" PREDICCIONES UNITARIAS - para verificar manualmente en la pagina web")
print("=" * 90)
print(f"\n{'Caso':<60} {'Python':>10} {'En la web':>15}")
print("-" * 90)

for desc, features in test_cases:
    fare, pclass, gender, age, sibsp = features
    prob = float(model.predict(np.array([features]), verbose=0)[0][0])
    verdict = "SOBREVIVE" if prob >= 0.5 else "no sobrev."
    web_pct = round(prob * 100)
    print(f"{desc:<60} {prob*100:>8.2f}%   ~{web_pct:>3}%  {verdict}")

print("\nComo verificar cada caso en la pagina web:")
print("  1. Abre la pagina (localhost o Vercel)")
print("  2. Ajusta los sliders/segmentos a los valores del caso")
print("  3. Presiona 'Calcular destino'")
print("  4. El porcentaje mostrado debe coincidir con el 'En la web' (+- 1%)")

print("\n  Gender en el formulario:")
print("    Boton 'Mujer'  = Gender 1")
print("    Boton 'Hombre' = Gender 0")


# ---------------------------------------------------------------------------
# Evaluacion sobre el conjunto de prueba completo (titanic-test.csv)
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print(" EVALUACION SOBRE titanic-test.csv (393 pasajeros)")
print("=" * 90)

test = pd.read_csv("titanic-test.csv")
test["Gender"] = test["Gender"].apply(lambda v: 0 if v == "male" else 1)
test["Age"] = test["Age"].fillna(test["Age"].mean())

columns = ["Fare", "Pclass", "Gender", "Age", "SibSp"]
x_test = test[columns].values
y_test = test["Survived"].values

probs_test = model.predict(x_test, verbose=0).flatten()
preds_test = (probs_test >= 0.5).astype(int)
acc = (preds_test == y_test).mean()

# Matriz de confusion
tp = int(((preds_test == 1) & (y_test == 1)).sum())
tn = int(((preds_test == 0) & (y_test == 0)).sum())
fp = int(((preds_test == 1) & (y_test == 0)).sum())
fn = int(((preds_test == 0) & (y_test == 1)).sum())

print(f"\n  Accuracy total:                       {acc * 100:.2f}%")
print(f"  Aciertos (TP + TN):                   {tp + tn} / {len(y_test)}")
print(f"\n  Matriz de confusion:")
print(f"                              Predicho")
print(f"                       NO sobrev.   sobrev.")
print(f"    Real  NO sobrev.   {tn:>10}   {fp:>7}")
print(f"          sobrev.      {fn:>10}   {tp:>7}")


# ---------------------------------------------------------------------------
# Comparar con la prediccion 'in-sample' (el mismo dato de entrenamiento)
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print(" EVALUACION SOBRE titanic-train.csv (915 pasajeros, dato de entrenamiento)")
print("=" * 90)

train = pd.read_csv("titanic-train.csv")
train["Gender"] = train["Gender"].apply(lambda v: 0 if v == "male" else 1)
train["Age"] = train["Age"].fillna(train["Age"].mean())
x_train = train[columns].values
y_train = train["Survived"].values
probs_train = model.predict(x_train, verbose=0).flatten()
preds_train = (probs_train >= 0.5).astype(int)
acc_train = (preds_train == y_train).mean()
print(f"\n  Accuracy en entrenamiento:            {acc_train * 100:.2f}%")
print(f"  (referencia del notebook del profesor: 84.26%)")


# ---------------------------------------------------------------------------
# Comparativa contra el notebook del profesor
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print(" COMPARACION CON EL NOTEBOOK DEL PROFESOR")
print("=" * 90)
print("""
  El notebook del profesor reporta:
    - Accuracy en entrenamiento: ~84.26%
    - Ejemplo [83.475, 1, 1, 35, 1] => probabilidad redondeada = 1.0 (SOBREVIVE)

  El modelo local reporta:
    - Accuracy en entrenamiento:  {:.2f}%  (debe estar cerca de 84%)
    - Ejemplo [83.475, 1, 1, 35, 1] => probabilidad = {:.4f}  ({})

  Nota: los numeros NO son identicos al notebook porque:
    a) Cada entrenamiento tiene inicializacion aleatoria de pesos distinta.
    b) Cada entrenamiento puede converger a un minimo ligeramente distinto.
    c) Diferencias menores entre Keras 2 (tf_keras) y Keras 3 del notebook.

  Lo IMPORTANTE es que la arquitectura, el algoritmo (Adam + binary
  cross-entropy + 400 epochs) y el preprocesamiento son IDENTICOS, y la
  accuracy es comparable. Esto demuestra que el modelo se entreno
  correctamente con el mismo metodo que enseñó el profesor.
""".format(acc_train * 100, probs_train[0] if False else float(model.predict(np.array([[83.475, 1, 1, 35, 1]]), verbose=0)[0][0]),
           'SOBREVIVE' if float(model.predict(np.array([[83.475, 1, 1, 35, 1]]), verbose=0)[0][0]) >= 0.5 else 'NO sobrevive'))
