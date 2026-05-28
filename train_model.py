"""
Entrenamiento local del modelo Titanic con Keras.

Replica exactamente el flujo del notebook Titanic_RN.ipynb del profesor,
y al final convierte el .h5 al formato TensorFlow.js para usar en el navegador.

REQUISITOS:
    Python 3.10 - 3.13 (TensorFlow NO soporta Python 3.14 todavia)
    Si tienes solo Python 3.14, instala Python 3.12 en paralelo:
    https://www.python.org/downloads/release/python-31210/
    Y usa: py -3.12 train_model.py

USO:
    py -3.12 -m pip install -r requirements.txt
    py -3.12 train_model.py

SALIDA:
    mimodelo.json                          (estructura, entregable Keras del profe)
    mimodelo.weights.h5                    (pesos, entregable Keras del profe)
    public/model/model.json + .bin         (mismo modelo, formato browser)
"""

import os
import sys
import shutil
import numpy as np
import pandas as pd

# Forzamos a TensorFlow a usar Keras 2 (tf_keras) en lugar de Keras 3.
# Razon: el convertidor tensorflowjs 4.22 todavia espera el formato Keras 2
# del modelo (ej. 'batch_input_shape' en vez del nuevo 'batch_shape'), asi
# que entrenar con tf_keras evita problemas de compatibilidad al convertir.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

try:
    import tf_keras as keras
    from tf_keras import layers
    from tf_keras.models import model_from_json
except ImportError:
    print("\n[ERROR] No se encontro Keras/TensorFlow.")
    print(f"Python en uso: {sys.version}")
    print("\nSi tu Python es 3.14, instala Python 3.12 primero:")
    print("  https://www.python.org/downloads/release/python-31210/")
    print("\nLuego:  py -3.12 -m pip install -r requirements.txt")
    print("Y:     py -3.12 train_model.py")
    sys.exit(1)


# ---------------------------------------------------------------------------
# 1. Cargar el dataset
# ---------------------------------------------------------------------------
print("=" * 70)
print(" ENTRENAMIENTO DEL MODELO TITANIC - Sesion 6 (Keras local)")
print("=" * 70)
print(f"\nPython: {sys.version.split()[0]}  |  Keras: {keras.__version__}")

print("\n[1/7] Cargando titanic-train.csv ...")
training = pd.read_csv("titanic-train.csv")
print(f"  -> {len(training)} registros cargados")


# ---------------------------------------------------------------------------
# 2. Preprocesamiento (identico al notebook del profesor)
# ---------------------------------------------------------------------------
print("\n[2/7] Preprocesando datos (igual al notebook) ...")

# Gender: male -> 0, female -> 1
training["Gender"] = training["Gender"].apply(
    lambda toLabel: 0 if toLabel == "male" else 1
)

# Age: rellenar NaN con la media
age_mean = training["Age"].mean()
training["Age"] = training["Age"].fillna(age_mean)
print(f"  -> Age NaN rellenados con la media: {age_mean:.4f}")


# ---------------------------------------------------------------------------
# 3. Definir entradas y objetivo
# ---------------------------------------------------------------------------
print("\n[3/7] Separando features y target ...")
columns = ["Fare", "Pclass", "Gender", "Age", "SibSp"]
x_input = training[list(columns)].values
y_target = training["Survived"].values
print(f"  -> features: {columns}")
print(f"  -> shape entrada: {x_input.shape}")


# ---------------------------------------------------------------------------
# 4. Construir la red neuronal (arquitectura del profesor)
# ---------------------------------------------------------------------------
print("\n[4/7] Construyendo red 5 -> 32 -> 32 -> 1 (ReLU + sigmoid) ...")
model = keras.Sequential()
model.add(layers.Dense(32, input_dim=5, activation="relu"))
model.add(layers.Dense(32, activation="relu", name="layer1"))
model.add(layers.Dense(1, activation="sigmoid", name="layer2"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
print()
model.summary()


# ---------------------------------------------------------------------------
# 5. Entrenar 400 epochs (igual al notebook)
# ---------------------------------------------------------------------------
print("\n[5/7] Entrenando 400 epochs (toma 1-3 minutos) ...")
model.fit(x_input, y_target, epochs=400, verbose=0)
score = model.evaluate(x_input, y_target, verbose=0)
print(f"\n  >>> Accuracy en entrenamiento: {score[1] * 100:.2f}%")

if os.path.exists("titanic-test.csv"):
    print("\n  Evaluando con titanic-test.csv ...")
    test = pd.read_csv("titanic-test.csv")
    test["Gender"] = test["Gender"].apply(
        lambda toLabel: 0 if toLabel == "male" else 1
    )
    test["Age"] = test["Age"].fillna(test["Age"].mean())
    x_test = test[list(columns)].values
    y_test = test["Survived"].values
    test_score = model.evaluate(x_test, y_test, verbose=0)
    print(f"  >>> Accuracy en prueba:        {test_score[1] * 100:.2f}%")


# ---------------------------------------------------------------------------
# 6. Guardar el modelo en formato Keras (entregable del profesor)
# ---------------------------------------------------------------------------
print("\n[6/7] Guardando modelo en formato Keras ...")

model_json = model.to_json()
with open("mimodelo.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("mimodelo.weights.h5")
print("  -> mimodelo.json")
print("  -> mimodelo.weights.h5")

# Verificacion: recargar y predecir
print("\n  Verificacion: cargando .h5 y prediciendo ...")
with open("mimodelo.json", "r") as f:
    loaded_model = model_from_json(f.read())
loaded_model.load_weights("mimodelo.weights.h5")
loaded_model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
)

ejemplo = np.array([[83.475, 1, 1, 35, 1]])  # mujer 35 anos, 1ra clase
prob = loaded_model.predict(ejemplo, verbose=0)[0][0]
print(f"  -> ejemplo (mujer 35, 1ra clase, £83.475, 1 familiar)")
print(f"  -> probabilidad: {prob:.4f}  ==> {'SOBREVIVE' if prob >= 0.5 else 'NO SOBREVIVE'}")


# ---------------------------------------------------------------------------
# 7. Convertir el .h5 a formato TensorFlow.js (para el navegador)
# ---------------------------------------------------------------------------
print("\n[7/7] Convirtiendo .h5 -> TensorFlow.js (para el navegador) ...")

try:
    import tensorflowjs as tfjs
except ImportError:
    print("\n  [!] tensorflowjs no esta instalado.")
    print("      py -3.12 -m pip install tensorflowjs")
    sys.exit(1)

out_dir = os.path.join("public", "model")
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir, exist_ok=True)

# Convertimos el modelo ya cargado desde el .h5 -- usa los mismos pesos.
tfjs.converters.save_keras_model(loaded_model, out_dir)

# Listar archivos generados
print(f"\n  Archivos generados en {out_dir}/:")
for f in sorted(os.listdir(out_dir)):
    size = os.path.getsize(os.path.join(out_dir, f))
    print(f"    {f}  ({size / 1024:.1f} KB)")


# ---------------------------------------------------------------------------
# Final
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(" ENTRENAMIENTO COMPLETADO")
print("=" * 70)
print("\nArchivos generados:")
print("  mimodelo.json                   <- entregable Keras")
print("  mimodelo.weights.h5             <- entregable Keras")
print("  public/model/model.json         <- mismo modelo, formato browser")
print("  public/model/group1-shard1of1.bin")
print("\nAhora puedes:")
print("  - Servir local:   py -3.12 -m http.server --directory public 8000")
print("  - O con tu 3.14:  python -m http.server --directory public 8000")
print("  - Y abrir:        http://localhost:8000")
print("  - O desplegar:    vercel --prod")
