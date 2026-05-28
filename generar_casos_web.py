"""
Genera una bateria de casos para probar en la web.
Toma pasajeros reales del dataset del profesor + el caso del notebook.

USO:
    py -3.12 generar_casos_web.py
"""
import os
import numpy as np
import pandas as pd

os.environ["TF_USE_LEGACY_KERAS"] = "1"
from tf_keras.models import model_from_json


with open("mimodelo.json") as f:
    model = model_from_json(f.read())
model.load_weights("mimodelo.weights.h5")
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


def fmt(case):
    """Format a test case as a row: [fare, pclass, gender, age, sibsp]"""
    feats = np.array([case["feats"]])
    prob = float(model.predict(feats, verbose=0)[0][0])
    pred = "SOBREVIVE" if prob >= 0.5 else "muere"
    fare, pclass, gender, age, sibsp = case["feats"]
    sex_label = "Mujer" if gender == 1 else "Hombre"
    class_label = ["1ra", "2da", "3ra"][int(pclass) - 1]
    real = case.get("real")
    real_label = "—" if real is None else ("SOBREVIVE" if real == 1 else "muere")
    match = "" if real is None else (" OK" if (prob >= 0.5) == (real == 1) else " XX")
    return {
        "name": case["name"],
        "sex": sex_label,
        "age": int(age),
        "pclass": class_label,
        "fare": fare,
        "sibsp": int(sibsp),
        "prob": prob,
        "pred": pred,
        "real": real_label,
        "match": match,
    }


# ---------------------------------------------------------------------------
# 1. El caso del notebook del profesor (cell 28)
# ---------------------------------------------------------------------------
cases = [
    {
        "name": "Caso del notebook (cell 28)",
        "feats": [83.475, 1, 1, 35, 1],
        "real": None,  # no hay outcome real, es un caso artificial
    },
]


# ---------------------------------------------------------------------------
# 2. Pasajeros REALES del dataset titanic-train.csv
# ---------------------------------------------------------------------------
train = pd.read_csv("titanic-train.csv")
train_proc = train.copy()
train_proc["Gender"] = train_proc["Gender"].apply(lambda v: 0 if v == "male" else 1)
age_mean = train_proc["Age"].mean()
train_proc["Age"] = train_proc["Age"].fillna(age_mean)

# Vamos a elegir manualmente 8 pasajeros representativos:
# - Mujer joven 1ra clase (alta probabilidad de sobrevivir)
# - Hombre joven 3ra clase, tarifa baja (probable muerte)
# - Nino pequeno 2da clase
# - Mujer mayor 1ra clase con familia
# - Hombre mayor 1ra clase
# - Mujer 2da clase mediana
# - Hombre joven 1ra clase
# - Hombre con familia grande en 3ra clase
indices_train = [3, 28, 100, 200, 300, 400, 500, 600]
for idx in indices_train:
    row = train_proc.iloc[idx]
    name = train.iloc[idx]["Name"]
    short_name = name.split(",")[0] + " (" + name.split(",")[1].split(".")[0].strip() + ".)"
    cases.append({
        "name": f"{short_name}",
        "feats": [
            float(row["Fare"]),
            int(row["Pclass"]),
            int(row["Gender"]),
            float(row["Age"]),
            int(row["SibSp"]),
        ],
        "real": int(row["Survived"]),
    })


# ---------------------------------------------------------------------------
# 3. Imprimir la tabla
# ---------------------------------------------------------------------------
print()
print("=" * 116)
print("  CASOS PARA PROBAR EN LA PAGINA WEB")
print("=" * 116)
print()
print(f"  {'Pasajero':<32} {'Sexo':<7} {'Edad':>5} {'Clase':>6} {'Tarifa':>10} {'Familia':>8}  |  {'Modelo':>10} {'Predice':>11}  {'Real':>11}")
print("  " + "-" * 114)

for case in cases:
    r = fmt(case)
    name_short = r["name"][:30]
    print(f"  {name_short:<32} {r['sex']:<7} {r['age']:>5} {r['pclass']:>6} {r['fare']:>8.4f} GBP {r['sibsp']:>5}  | "
          f" {r['prob']*100:>8.1f}% {r['pred']:>11}{r['match']:<3}  {r['real']:>11}")

print()
print("=" * 116)
print()
print("  Como probarlo en la web:")
print("    1. Abre la pagina (local o el deploy de Vercel)")
print("    2. Para cada fila, ajusta:")
print("       - Sexo:           boton 'Mujer' o 'Hombre'")
print("       - Edad:           slider hasta el numero indicado")
print("       - Clase:          boton 1ra/2da/3ra")
print("       - Tarifa:         escribe el valor exacto en el input £")
print("       - Familia:        stepper hasta el numero indicado")
print("    3. Aprieta 'Calcular destino'")
print("    4. El % mostrado debe coincidir con la columna 'Modelo' +- 1%")
print()
print("  Como interpretar:")
print("    'Modelo'  = probabilidad que predice tu .h5 (mostrado en el ring de la web)")
print("    'Predice' = SOBREVIVE si prob >= 50%, muere si < 50%  (lo mismo que la web)")
print("    'Real'    = lo que realmente paso historicamente (columna Survived del CSV)")
print("    ✓ / ✗     = si la prediccion del modelo coincide con el outcome historico")
print()
print(f"  Este modelo tiene ~83% accuracy global, asi que de 8 casos esperarias ~1-2 fallos")
print(f"  (marcados con ✗).  La unica garantia exacta es el caso del notebook (cell 28).")
print()
