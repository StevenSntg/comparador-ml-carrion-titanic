import numpy as np
import pandas as pd
import pytest
from carrion_preprocess import (
    DEPARTAMENTOS, FEATURES, edad_a_anios, encode_sexo,
    encode_departamento, build_xy,
)


def test_departamentos_son_19_ordenados():
    assert len(DEPARTAMENTOS) == 19
    assert DEPARTAMENTOS[0] == "AMAZONAS"
    assert DEPARTAMENTOS[1] == "ANCASH"
    assert DEPARTAMENTOS[18] == "UCAYALI"


def test_features_son_cinco():
    assert FEATURES == ["edad", "sexo", "departamento", "anio", "semana"]


def test_edad_en_anios_se_mantiene():
    assert edad_a_anios(33, "A") == 33.0


def test_edad_en_meses_se_divide_por_12():
    assert edad_a_anios(12, "M") == 1.0
    assert edad_a_anios(6, "M") == 0.5


def test_edad_en_dias_se_divide_por_365():
    assert edad_a_anios(365, "D") == 1.0


def test_encode_sexo():
    assert encode_sexo("M") == 0
    assert encode_sexo("F") == 1


def test_encode_departamento_alfabetico():
    assert encode_departamento("AMAZONAS") == 0
    assert encode_departamento("LIMA") == 10
    assert encode_departamento("UCAYALI") == 18


def test_build_xy_devuelve_5_columnas_y_target_binario():
    df = pd.DataFrame({
        "departamento": ["ANCASH", "LIMA"],
        "edad": [20, 6],
        "tipo_edad": ["A", "M"],
        "sexo": ["M", "F"],
        "ano": [2004, 2010],
        "semana": [31, 4],
        "enfermedad": ["ENFERMEDAD DE CARRION AGUDA",
                       "ENFERMEDAD DE CARRION ERUPTIVA"],
    })
    X, y = build_xy(df)
    assert X.shape == (2, 5)
    assert list(y) == [0, 1]
    # fila 0: edad 20 años, sexo M=0, ANCASH=1, 2004, semana 31
    assert list(X[0]) == [20.0, 0, 1, 2004, 31]
    # fila 1: 6 meses = 0.5, sexo F=1, LIMA=10
    assert list(X[1]) == [0.5, 1, 10, 2010, 4]
