"""Verifica la paridad entre los modelos Python y los artefactos que consume
la web, para los árboles de decisión de ambos casos (Titanic y Carrión).

- AD: recorre el ad.json con la misma lógica que el JS (export_tree.predict_dict)
  y compara contra el .pkl de sklearn sobre el conjunto de prueba completo.
- La RN usa los mismos pesos del .h5 convertidos a TF.js (sin reentrenar),
  por lo que su paridad está garantizada por construcción.

USO:
    py -3.12 verificar_predicciones.py
"""
import json
import pickle
import pandas as pd

from export_tree import predict_dict
from carrion_preprocess import build_xy


def verif_arbol_carrion():
    with open("miarbolcarrion.pkl", "rb") as f:
        clf = pickle.load(f)
    d = json.load(open("public/models/carrion/ad.json", encoding="utf-8"))
    X, _ = build_xy(pd.read_csv("data/carrion-test.csv"))
    sk = clf.predict(X)
    iguales = sum(predict_dict(d, X[i])[0] == sk[i] for i in range(len(X)))
    print(f"[Arbol Carrion] {iguales}/{len(X)} coinciden (web == sklearn)")
    assert iguales == len(X)


def verif_arbol_titanic():
    with open("miarboltitanic.pkl", "rb") as f:
        clf = pickle.load(f)
    d = json.load(open("public/models/titanic/ad.json", encoding="utf-8"))
    df = pd.read_csv("titanic-test.csv")
    df["Gender"] = df["Gender"].apply(lambda g: 0 if g == "male" else 1)
    df["Age"] = df["Age"].fillna(df["Age"].mean())
    X = df[["Fare", "Pclass", "Gender", "Age", "SibSp"]].values
    sk = clf.predict(X)
    iguales = sum(predict_dict(d, X[i])[0] == sk[i] for i in range(len(X)))
    print(f"[Arbol Titanic] {iguales}/{len(X)} coinciden (web == sklearn)")
    assert iguales == len(X)


if __name__ == "__main__":
    verif_arbol_titanic()
    verif_arbol_carrion()
    print("\nParidad de arboles verificada. (La RN usa los mismos pesos .h5/TF.js.)")
