"""Entrena los dos modelos del caso CLÍNICO de Carrión (síntomas) y los exporta
para la web. Dataset sintético (ver generar_carrion_clinico.py).

  - Árbol de Decisión -> public/models/carrion_clinico/ad.json (+ .pkl)
  - Red Neuronal 8->32->32->1 -> public/models/carrion_clinico/rn/ (TF.js)

Los síntomas ya son binarios (0/1), así que NO se normaliza.
"""
import os, json, shutil, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

from export_tree import tree_to_dict, save_json

SINTOMAS = ["fiebre_alta", "palidez_anemia", "fatiga_debilidad", "dolor_cabeza_cuerpo",
            "ictericia", "ganglios_hinchazon", "verrugas_piel", "sangrado_lesiones"]
CLASS_LABELS = ["AGUDA", "ERUPTIVA"]
OUT = "public/models/carrion_clinico"

df = pd.read_csv("data/carrion_clinico.csv")
X = df[SINTOMAS].values.astype(float)
y = (df["fase"] == "ERUPTIVA").astype(int).values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
os.makedirs(OUT, exist_ok=True)

# ─── Árbol de Decisión ────────────────────────────────────────────────────
print("[Árbol] entrenando ...")
clf = DecisionTreeClassifier(criterion="entropy", random_state=42).fit(Xtr, ytr)
print(f"  train={clf.score(Xtr,ytr)*100:.2f}%  test={clf.score(Xte,yte)*100:.2f}%")
print(confusion_matrix(yte, clf.predict(Xte)))
print(classification_report(yte, clf.predict(Xte), target_names=CLASS_LABELS))
pickle.dump(clf, open("miarbolclinico.pkl", "wb"))
save_json(tree_to_dict(clf, SINTOMAS, CLASS_LABELS), f"{OUT}/ad.json")
print(f"  -> {OUT}/ad.json")

# ─── Red Neuronal ─────────────────────────────────────────────────────────
print("\n[RN] entrenando (8 -> 32 -> 32 -> 1) ...")
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tf_keras as keras
from tf_keras import layers
from tf_keras.models import model_from_json

model = keras.Sequential()
model.add(layers.Dense(32, input_dim=len(SINTOMAS), activation="relu"))
model.add(layers.Dense(32, activation="relu", name="layer1"))
model.add(layers.Dense(1, activation="sigmoid", name="layer2"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(Xtr, ytr, epochs=200, batch_size=64, verbose=0)
print(f"  train={model.evaluate(Xtr,ytr,verbose=0)[1]*100:.2f}%  "
      f"test={model.evaluate(Xte,yte,verbose=0)[1]*100:.2f}%")

with open("mimodeloclinico.json", "w") as f:
    f.write(model.to_json())
model.save_weights("mimodeloclinico.weights.h5")
with open("mimodeloclinico.json") as f:
    loaded = model_from_json(f.read())
loaded.load_weights("mimodeloclinico.weights.h5")

import tensorflowjs as tfjs
rn_out = f"{OUT}/rn"
if os.path.exists(rn_out):
    shutil.rmtree(rn_out)
os.makedirs(rn_out, exist_ok=True)
tfjs.converters.save_keras_model(loaded, rn_out)
print(f"  -> {rn_out}/model.json")
