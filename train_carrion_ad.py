"""Entrena el Árbol de Decisión para Carrión y exporta a JSON para la web.

Usa el MISMO preprocesamiento que la Red Neuronal (carrion_preprocess),
cumpliendo el requisito de 'misma fuente de datos para ambos modelos'.
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

from carrion_preprocess import build_xy, FEATURES, CLASS_LABELS
from export_tree import tree_to_dict, save_json

print("[1/5] Cargando datos de entrenamiento y prueba ...")
train = pd.read_csv("data/carrion-train.csv")
test = pd.read_csv("data/carrion-test.csv")
X_train, y_train = build_xy(train)
X_test, y_test = build_xy(test)
print(f"  train={X_train.shape}  test={X_test.shape}")

print("[2/5] Entrenando DecisionTreeClassifier(criterion='entropy') ...")
clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
clf.fit(X_train, y_train)

acc_train = clf.score(X_train, y_train) * 100
acc_test = clf.score(X_test, y_test) * 100
print(f"  Accuracy train: {acc_train:.2f}%")
print(f"  Accuracy test:  {acc_test:.2f}%")

print("[3/5] Metricas en test ...")
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=CLASS_LABELS))

print("[4/5] Guardando modelo (.pkl) ...")
with open("miarbolcarrion.pkl", "wb") as f:
    pickle.dump(clf, f)

print("[5/5] Exportando a JSON para la web ...")
os.makedirs("public/models/carrion", exist_ok=True)
d = tree_to_dict(clf, FEATURES, CLASS_LABELS)
save_json(d, "public/models/carrion/ad.json")
print("  -> public/models/carrion/ad.json")
