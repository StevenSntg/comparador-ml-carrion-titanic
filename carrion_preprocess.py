"""Preprocesamiento compartido para la enfermedad de Carrión.

Usado por los notebooks y los scripts de entrenamiento para garantizar
que RN y AD reciben EXACTAMENTE el mismo preprocesamiento (requisito del
examen: misma fuente de datos para ambos modelos).
"""
import numpy as np
import pandas as pd

# 19 departamentos en orden alfabético (= LabelEncoder de sklearn).
DEPARTAMENTOS = [
    "AMAZONAS", "ANCASH", "APURIMAC", "AYACUCHO", "CAJAMARCA", "CUSCO",
    "HUANUCO", "JUNIN", "LA LIBERTAD", "LAMBAYEQUE", "LIMA", "LORETO",
    "MADRE DE DIOS", "PASCO", "PIURA", "PUNO", "SAN MARTIN", "TACNA", "UCAYALI",
]
_DEPTO_INDEX = {nombre: i for i, nombre in enumerate(DEPARTAMENTOS)}

FEATURES = ["edad", "sexo", "departamento", "anio", "semana"]

TARGET_MAP = {
    "ENFERMEDAD DE CARRION AGUDA": 0,
    "ENFERMEDAD DE CARRION ERUPTIVA": 1,
}
CLASS_LABELS = ["AGUDA", "ERUPTIVA"]


def edad_a_anios(edad, tipo_edad):
    """Convierte la edad a años según tipo_edad: A=años, M=meses, D=días."""
    t = str(tipo_edad).strip().upper()
    if t == "M":
        return edad / 12.0
    if t == "D":
        return edad / 365.0
    return float(edad)


def encode_sexo(sexo):
    return 0 if str(sexo).strip().upper() == "M" else 1


def encode_departamento(nombre):
    return _DEPTO_INDEX[str(nombre).strip().upper()]


def load_raw(path):
    """Carga el CSV crudo del MINSA."""
    return pd.read_csv(path)


def build_xy(df):
    """Devuelve (X, y) numpy listos para entrenar.

    X: matriz (n, 5) con FEATURES en orden.
    y: vector binario (0=AGUDA, 1=ERUPTIVA).
    """
    df = df.copy()
    df["edad_anios"] = df.apply(
        lambda r: edad_a_anios(r["edad"], r["tipo_edad"]), axis=1
    )
    df["sexo_cod"] = df["sexo"].apply(encode_sexo)
    df["depto_cod"] = df["departamento"].apply(encode_departamento)
    df["target"] = df["enfermedad"].map(TARGET_MAP)
    df = df.dropna(subset=["target"])  # descarta enfermedades fuera del mapa

    X = df[["edad_anios", "sexo_cod", "depto_cod", "anio", "semana"]].values.astype(float)
    y = df["target"].astype(int).values
    return X, y
