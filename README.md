# Comparador de Modelos ML — Titanic & Enfermedad de Carrión

Página web que **compara una Red Neuronal y un Árbol de Decisión** sobre dos casos
de clasificación binaria, usando para cada caso una única fuente de datos compartida
por ambos modelos:

1. **Titanic** — ¿sobrevive el pasajero? (datos históricos del naufragio de 1912)
2. **Enfermedad de Carrión** — ¿fase AGUDA o ERUPTIVA? (datos abiertos del MINSA, 2000–2024)

**Curso:** Inteligencia Artificial / Sistemas Inteligentes — Sesión 9 — UNTELS 2026

---

## Vista rápida

- **Frontend:** HTML + React 18 (CDN) + TensorFlow.js 4.20 + Babel standalone
- **Comparación lado a lado:** un solo formulario por caso; al enviar, RN y Árbol
  predicen a la vez y se muestran juntos con su veredicto y confianza.
- **Red Neuronal:** Keras `Sequential(5 → 32 → 32 → 1)`, Adam + binary cross-entropy,
  400 epochs. En Carrión se normaliza (min-max) y se guarda `norm.json` para el navegador.
- **Árbol de Decisión:** scikit-learn `DecisionTreeClassifier(criterion="entropy")`,
  exportado a JSON y recorrido en el navegador con ~20 líneas de JS (sin dependencias).
- **Hosting:** Vercel (estático, sin backend).

## Casos y datos

| Caso | Fuente | Filas | Target | 5 features |
|---|---|---|---|---|
| Titanic | `titanic-train/test.csv` | 915 / 393 | Survived (0/1) | Fare, Pclass, Gender, Age, SibSp |
| Carrión | MINSA `data/carrion_raw.csv` | 46,121 | AGUDA(0) / ERUPTIVA(1) | edad, sexo, departamento, año, semana |

## Flujo de entrenamiento (genera los modelos de `public/models/`)

```powershell
# 1. Una sola vez: instalar dependencias
py -3.12 -m pip install -r requirements.txt
py -3.12 -m pip install --no-deps tensorflowjs

# 2. Carrión: split + ambos modelos
py -3.12 make_split.py            # data/carrion-train.csv + carrion-test.csv
py -3.12 train_carrion_ad.py      # public/models/carrion/ad.json
py -3.12 train_carrion_rn.py      # public/models/carrion/rn/ + norm.json

# 3. Titanic: red neuronal (existente) + árbol nuevo
py -3.12 train_model.py           # public/models/titanic/rn/  (RN, ya provista)
py -3.12 train_titanic_ad.py      # public/models/titanic/ad.json

# 4. Verificar paridad web == Python y correr tests
py -3.12 verificar_predicciones.py
py -3.12 -m pytest -q
node test_predict_tree.mjs
```

> Nota: `train_model.py` (RN del Titanic) escribe en `public/model/`. Tras ejecutarlo,
> mover su salida a `public/models/titanic/rn/` (ver estructura abajo).

## Servir y desplegar

```powershell
py -3.12 -m http.server --directory public 8123   # http://localhost:8123
vercel --prod                                     # despliegue estático
```

## Estructura

```
comparador-ml/
├── carrion_preprocess.py        # limpieza/encoding compartido (RN y AD)
├── make_split.py                # split estratificado de Carrión
├── export_tree.py               # árbol sklearn -> JSON (+ recorrido de referencia)
├── train_carrion_ad.py / _rn.py # entrenan y exportan los modelos de Carrión
├── train_titanic_ad.py          # entrena y exporta el árbol del Titanic
├── train_model.py               # RN del Titanic (de la Sesión 6)
├── verificar_predicciones.py    # paridad web == sklearn (árboles)
├── test_*.py / test_*.mjs       # tests (preprocesamiento, paridad)
└── public/
    ├── index.html               # SPA: tabs de caso + comparación lado a lado
    └── models/
        ├── titanic/{rn/, ad.json}
        └── carrion/{rn/, ad.json, norm.json}
```

## Métricas de referencia (entrenamiento)

| Caso | Modelo | Accuracy train | Accuracy test |
|---|---|---|---|
| Titanic | Árbol de Decisión | ~97% | ~74% |
| Titanic | Red Neuronal | ~84% | ~76% |
| Carrión | Árbol de Decisión | ~93% | ~67% |
| Carrión | Red Neuronal | ~73% | ~73% |

En Carrión la Red Neuronal generaliza de forma estable (train ≈ test), mientras que
el Árbol memoriza el entrenamiento (overfitting). Esta diferencia es el eje de la
comparativa del informe.

## Notebooks entregables

En la carpeta superior (`Sesion8/`): `Carrion_RN.ipynb` y `Carrion_AD.ipynb`,
que entrenan cada modelo y muestran accuracy, matriz de confusión y métricas.
