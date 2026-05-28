"""Entrena el Árbol de Decisión del Titanic y lo exporta a JSON para la web.
Replica el preprocesamiento del notebook Titanic_AD.ipynb del profesor.
"""
import os
import pickle
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from export_tree import tree_to_dict, save_json

COLUMNS = ["Fare", "Pclass", "Gender", "Age", "SibSp"]
CLASS_LABELS = ["No sobrevive", "Sobrevive"]


def preprocess(df):
    df = df.copy()
    df["Gender"] = df["Gender"].apply(lambda g: 0 if g == "male" else 1)
    df["Age"] = df["Age"].fillna(df["Age"].mean())
    return df


train = preprocess(pd.read_csv("titanic-train.csv"))
test = preprocess(pd.read_csv("titanic-test.csv"))
X, Y = train[COLUMNS].values, train["Survived"].values
Xt, Yt = test[COLUMNS].values, test["Survived"].values

clf = DecisionTreeClassifier(criterion="entropy", random_state=42).fit(X, Y)
print(f"Accuracy train: {clf.score(X, Y)*100:.2f}%")
print(f"Accuracy test:  {clf.score(Xt, Yt)*100:.2f}%")
print(confusion_matrix(Yt, clf.predict(Xt)))
print(classification_report(Yt, clf.predict(Xt), target_names=CLASS_LABELS))

with open("miarboltitanic.pkl", "wb") as f:
    pickle.dump(clf, f)
os.makedirs("public/models/titanic", exist_ok=True)
save_json(tree_to_dict(clf, COLUMNS, CLASS_LABELS), "public/models/titanic/ad.json")
print("-> public/models/titanic/ad.json")
