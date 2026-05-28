"""Genera carrion-train.csv (80%) y carrion-test.csv (20%), estratificado
por el tipo de enfermedad (AGUDA/ERUPTIVA)."""
import pandas as pd
from sklearn.model_selection import train_test_split
from carrion_preprocess import TARGET_MAP

df = pd.read_csv("data/carrion_raw.csv")
df = df[df["enfermedad"].isin(TARGET_MAP.keys())].copy()

train, test = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df["enfermedad"]
)
train.to_csv("data/carrion-train.csv", index=False)
test.to_csv("data/carrion-test.csv", index=False)

print(f"train: {len(train)} filas  |  test: {len(test)} filas")
print("train target:\n", train["enfermedad"].value_counts())
print("test  target:\n", test["enfermedad"].value_counts())
