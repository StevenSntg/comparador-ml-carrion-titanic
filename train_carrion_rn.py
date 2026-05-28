"""Entrena la Red Neuronal para Carrión (misma arquitectura que el Titanic:
5 -> 32 -> 32 -> 1) con normalización min-max, y exporta a TensorFlow.js.

Guarda norm.json con los min/max por feature para aplicar la MISMA
normalización en el navegador.
"""
import os
import json
import shutil
import numpy as np
import pandas as pd

os.environ["TF_USE_LEGACY_KERAS"] = "1"  # igual que train_model.py del Titanic
import tf_keras as keras
from tf_keras import layers
from tf_keras.models import model_from_json

from carrion_preprocess import build_xy, FEATURES

print("[1/6] Cargando datos ...")
train = pd.read_csv("data/carrion-train.csv")
test = pd.read_csv("data/carrion-test.csv")
X_train, y_train = build_xy(train)
X_test, y_test = build_xy(test)

print("[2/6] Normalizacion min-max (guardada para la web) ...")
x_min = X_train.min(axis=0)
x_max = X_train.max(axis=0)
rng = np.where((x_max - x_min) == 0, 1, x_max - x_min)  # evita division por 0
Xtr = (X_train - x_min) / rng
Xte = (X_test - x_min) / rng

os.makedirs("public/models/carrion", exist_ok=True)
with open("public/models/carrion/norm.json", "w", encoding="utf-8") as f:
    json.dump({"feature_names": FEATURES,
               "min": x_min.tolist(), "max": x_max.tolist()}, f)

print("[3/6] Construyendo red 5 -> 32 -> 32 -> 1 ...")
model = keras.Sequential()
model.add(layers.Dense(32, input_dim=5, activation="relu"))
model.add(layers.Dense(32, activation="relu", name="layer1"))
model.add(layers.Dense(1, activation="sigmoid", name="layer2"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

print("[4/6] Entrenando 400 epochs ...")
model.fit(Xtr, y_train, epochs=400, verbose=0)
print(f"  Accuracy train: {model.evaluate(Xtr, y_train, verbose=0)[1]*100:.2f}%")
print(f"  Accuracy test:  {model.evaluate(Xte, y_test, verbose=0)[1]*100:.2f}%")

print("[5/6] Guardando modelo Keras ...")
with open("mimodelocarrion.json", "w") as f:
    f.write(model.to_json())
model.save_weights("mimodelocarrion.weights.h5")

with open("mimodelocarrion.json") as f:
    loaded = model_from_json(f.read())
loaded.load_weights("mimodelocarrion.weights.h5")
loaded.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

print("[6/6] Convirtiendo a TensorFlow.js ...")
import tensorflowjs as tfjs
out = "public/models/carrion/rn"
if os.path.exists(out):
    shutil.rmtree(out)
os.makedirs(out, exist_ok=True)
tfjs.converters.save_keras_model(loaded, out)
print("  -> public/models/carrion/rn/model.json")
