"""Genera un dataset CLÍNICO SIMULADO de la enfermedad de Carrión para un
formulario de síntomas (fines educativos).

⚠️  IMPORTANTE: estos datos NO provienen del MINSA ni de ninguna fuente real.
Son SINTÉTICOS, generados muestreando síntomas con probabilidades condicionadas
a cada fase, basadas en los criterios clínicos descritos en la literatura médica:

  - Fase AGUDA (Fiebre de la Oroya): fiebre alta, anemia hemolítica severa,
    fatiga, cefalea, ictericia, hepato/esplenomegalia y linfadenopatía.
  - Fase ERUPTIVA (Verruga Peruana): verrugas/nódulos rojizos en la piel que
    pueden sangrar; fase menos grave.

Salida: data/carrion_clinico.csv
"""
import numpy as np
import pandas as pd

RNG = np.random.RandomState(42)
N = 6000

# 8 síntomas (features) que un paciente puede reconocer/reportar.
SINTOMAS = [
    "fiebre_alta",        # >39°C con escalofríos
    "palidez_anemia",     # palidez / signos de anemia
    "fatiga_debilidad",   # fatiga intensa, debilidad
    "dolor_cabeza_cuerpo",# cefalea y malestar corporal
    "ictericia",          # piel/ojos amarillentos
    "ganglios_hinchazon", # ganglios inflamados / hígado-bazo agrandados
    "verrugas_piel",      # verrugas o nódulos rojizos/púrpuras
    "sangrado_lesiones",  # sangrado o molestia de esas lesiones
]

# P(síntoma = 1 | fase). Valores intermedios -> casos ambiguos (no trivial).
P_AGUDA = {
    "fiebre_alta": 0.92, "palidez_anemia": 0.90, "fatiga_debilidad": 0.88,
    "dolor_cabeza_cuerpo": 0.80, "ictericia": 0.55, "ganglios_hinchazon": 0.50,
    "verrugas_piel": 0.05, "sangrado_lesiones": 0.03,
}
P_ERUPTIVA = {
    "fiebre_alta": 0.12, "palidez_anemia": 0.16, "fatiga_debilidad": 0.35,
    "dolor_cabeza_cuerpo": 0.30, "ictericia": 0.03, "ganglios_hinchazon": 0.08,
    "verrugas_piel": 0.90, "sangrado_lesiones": 0.62,
}

# Mezcla de fases (la eruptiva/crónica es algo más frecuente entre los que
# llegan a consulta dermatológica; usamos una proporción equilibrada).
fases = RNG.choice(["AGUDA", "ERUPTIVA"], size=N, p=[0.5, 0.5])

filas = []
for fase in fases:
    P = P_AGUDA if fase == "AGUDA" else P_ERUPTIVA
    fila = {s: int(RNG.rand() < P[s]) for s in SINTOMAS}
    fila["fase"] = fase
    filas.append(fila)

df = pd.DataFrame(filas, columns=SINTOMAS + ["fase"])

# Ruido: en ~8% de los casos, voltea un síntoma al azar (errores de reporte /
# presentaciones atípicas) para que la clasificación no sea trivial.
n_ruido = int(0.08 * N)
idx = RNG.choice(N, size=n_ruido, replace=False)
for i in idx:
    s = RNG.choice(SINTOMAS)
    df.at[i, s] = 1 - df.at[i, s]

df.to_csv("data/carrion_clinico.csv", index=False)
print(f"Generado data/carrion_clinico.csv  ({len(df)} filas)")
print(df["fase"].value_counts())
print("\nPromedio de cada síntoma por fase:")
print(df.groupby("fase")[SINTOMAS].mean().round(2).T)
