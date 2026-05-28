# Predictor de Supervivencia del Titanic

Pagina web que predice si un pasajero del Titanic habria sobrevivido,
usando una red neuronal Keras entrenada con datos historicos.

**Curso:** Inteligencia Artificial — Sesion 6 — UNTELS 2026

---

## Vista rapida

- **Frontend:** HTML + React 18 (CDN) + TensorFlow.js 4.20 + diseno propio
- **Modelo:** Keras `Sequential(5 -> 32 -> 32 -> 1)`, Adam + binary cross-entropy, 400 epochs
- **Entrenamiento:** Python 3.12 + `tf_keras` (TensorFlow 2.21)
- **Hosting:** Vercel (static, sin backend)
- **Background:** Video atmosferico generado con Higgsfield (Veo 3.1 Lite)

## Como correrlo

Ver [`INSTRUCCIONES.md`](./INSTRUCCIONES.md) para el detalle completo. Resumen:

```powershell
# 1. Una sola vez: instalar y parchar dependencias
py -3.12 setup.py

# 2. Una sola vez: entrenar el modelo (genera .h5 + TF.js)
py -3.12 train_model.py

# 3. Servir local
python -m http.server --directory public 8000
# abre http://localhost:8000

# 4. Desplegar
vercel --prod
```

## Estructura

```
titanic-predictor/
├── setup.py                    # Instala TF, tf-keras, tensorflowjs + parches
├── train_model.py              # Entrena la red y convierte a TF.js
├── verificar_predicciones.py   # Compara predicciones browser vs Python
├── train_model.ipynb           # Alternativa: notebook para Colab
├── requirements.txt
├── titanic-train.csv           # 915 pasajeros (entrenamiento)
├── titanic-test.csv            # 393 pasajeros (evaluacion)
├── vercel.json / .vercelignore
├── INSTRUCCIONES.md            # Guia detallada
│
├── mimodelo.json               # Modelo Keras (entregable)
├── mimodelo.weights.h5         # Pesos Keras (entregable)
│
└── public/                     # Lo que despliega Vercel
    ├── index.html              # UI completa con TF.js
    ├── bg-ocean.mp4            # Video oceanico de fondo (6s, loop)
    └── model/
        ├── model.json          # Modelo en formato browser (mismos pesos del .h5)
        └── group1-shard1of1.bin
```

## Como las predicciones vienen del `.h5`

```
   train_model.py  (una vez)
   ┌─────────────────────────────────┐
   │ 1. Entrena Keras Sequential     │
   │ 2. Guarda mimodelo.weights.h5   │
   │ 3. Convierte a formato TF.js    │
   └─────────────────────────────────┘
              │
              ▼
   Browser  (cada visita)
   ┌─────────────────────────────────┐
   │ tf.loadLayersModel('model.json')│
   │ model.predict(input) -> prob.   │
   └─────────────────────────────────┘
```

La conversion NO reentrena. El modelo en el navegador tiene los **mismos pesos**
que el `.h5` entregado al profesor.

## Metricas

| Metrica | Valor |
|---|---|
| Accuracy en `titanic-train.csv` | ~83-85% |
| Accuracy en `titanic-test.csv`  | ~76-78% |
| Referencia del notebook         | 84.26% |

Las pequenas diferencias vs el notebook son por inicializacion aleatoria de pesos
y diferencias entre Keras 2 y Keras 3 — la arquitectura, algoritmo y datos son
identicos.
