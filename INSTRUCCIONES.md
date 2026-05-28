# Predictor de Supervivencia del Titanic

Pagina web que predice si un pasajero del Titanic habria sobrevivido, usando
una red neuronal Keras entrenada con los datos historicos del profesor.

---

## Como funciona

1. **`train_model.py`** entrena la red neuronal con Keras y produce:
   - `mimodelo.json` + `mimodelo.weights.h5`
   - `public/model/model.json` + `.bin` — **el mismo modelo** convertido al
     formato que el navegador puede leer (no se reentrena, solo se re-empaqueta)
2. **`public/index.html`** carga el modelo TF.js **una sola vez** al abrir
   la pagina. Cuando el usuario hace click en "Calcular destino", ejecuta
   `model.predict()` — **NO se reentrena nada**, son los **mismos pesos** que
   estan en el archivo `.h5`.

---

## Setup inicial (UNA VEZ)

### Paso 1 — Instalar Python 3.12

TensorFlow no soporta Python 3.14 todavia (maximo 3.13). Como tu sistema ya
tiene Python 3.14, instala 3.12 en paralelo (no rompe nada).

1. Descarga el instalador: <https://www.python.org/downloads/release/python-31210/>
   (Windows installer 64-bit, ~25 MB)
2. Al instalar, asegurate de marcar:
   - **"Add python.exe to PATH"**
   - **"py launcher"** (importante, asi puedes usar `py -3.12`)
3. Verifica que coexisten:

```powershell
py -3.12 --version    # debe decir Python 3.12.x
py -3.14 --version    # debe decir Python 3.14.x
```

### Paso 2 — Correr el setup automatico

```powershell
cd C:\Users\Usuario\Desktop\Sesion6\titanic-predictor
py -3.12 setup.py
```

Esto:
- Instala `tensorflow`, `tf-keras`, `pandas`, `numpy` (~500 MB la primera vez)
- Instala `tensorflowjs` con `--no-deps` (evita el conflicto con TFDF que pide TF 2.15)
- Fija `setuptools<81` (para que `tensorflow_hub` encuentre `pkg_resources`)
- Parcha 2 archivos de `tensorflowjs` para que los imports de TFDF y JAX sean opcionales

> Los parches son necesarios porque `tensorflowjs 4.22` importa `tensorflow_decision_forests`
> y `jax` al cargar, aunque solo necesitemos el convertidor Keras. TFDF no tiene wheels
> compatibles con TF 2.21 en Windows, asi que envolvemos esos imports en try/except.

Toma 2-3 minutos la primera vez. Es **idempotente** — si lo corres de nuevo, omite
los pasos que ya estan listos.

### Paso 3 — Entrenar el modelo

```powershell
py -3.12 train_model.py
```

Toma 1-3 minutos. Al final habras generado:

```
mimodelo.json                       (entregable Keras del profesor)
mimodelo.weights.h5                 (entregable Keras del profesor)
public/model/model.json             (mismo modelo, formato browser)
public/model/group1-shard1of1.bin
```

Resultado esperado: ~83-85% accuracy en entrenamiento, ~76-78% en prueba.

---

## Probar localmente

```powershell
# Sirve la carpeta public/. Puedes usar cualquiera de los dos Python:
python -m http.server --directory public 8000
# o:  py -3.12 -m http.server --directory public 8000
```

Abre <http://localhost:8000> en el navegador.

> Si abres `public/index.html` con doble click NO va a funcionar — el browser
> bloquea `fetch()` desde `file://`. Hay que usar el servidor de arriba.

---

## Verificar que la web predice IGUAL que el modelo entrenado

Para confirmar que la pagina web esta usando exactamente el `.h5` y no
inventa nada, hay un script que predice en Python (cargando el `.h5`) y
te lista varios casos para comparar contra el browser.

```powershell
py -3.12 verificar_predicciones.py
```

Imprime una tabla asi:

```
Caso                                                       Python    En la web
Ejemplo del notebook (mujer 35, 1ra clase, £83.475, 1)     96.81%    ~97%  SOBREVIVE
Mujer 28, 1ra clase, £71, 1 familiar (default de la web)   97.03%    ~97%  SOBREVIVE
Hombre 35, 3ra clase, £7.05, sin familiares                 7.28%    ~ 7%  no sobrev.
Nino 5 anos, 2da clase, £30, 2 hermanos                    23.59%    ~24%  no sobrev.
...
```

Tambien reporta:
- Accuracy sobre `titanic-test.csv` (~76-78%)
- Accuracy sobre `titanic-train.csv` (~83-85%, cerca del 84.26% del notebook)
- Matriz de confusion

**Como comparar contra la web:**
1. Corre `py -3.12 verificar_predicciones.py` → te imprime las predicciones de Python.
2. Abre la pagina web (local o desplegada).
3. Para cada caso, ajusta los sliders / botones al input correspondiente:
   - **Sexo** "Mujer" = Gender 1, "Hombre" = Gender 0
   - **Edad** slider en el valor exacto
   - **Clase** boton segun corresponda
   - **Tarifa** slider en el valor (en libras)
   - **Hermanos/conyuge** stepper en el valor
4. Aprieta "Calcular destino" y compara el % mostrado con el % de Python.

Deben coincidir dentro de **+/- 1%** (la diferencia es por redondeo de
floats entre TensorFlow Python y TensorFlow.js). Si difieren mas, algo
fallo en la conversion `.h5` -> `model.json`.

---

## Desplegar a Vercel

### Opcion A — Vercel CLI

```powershell
# Instala la CLI una vez:
npm i -g vercel

# Desde la carpeta titanic-predictor:
vercel              # primera vez (te hace preguntas; deja los defaults)
vercel --prod       # cuando quieras publicar la URL final
```

`vercel.json` ya tiene configurado `outputDirectory: public`, asi que SOLO
se sube `public/` (y NO los archivos `.h5`, `.csv`, ni los scripts Python — eso
lo maneja `.vercelignore`).

### Opcion B — Conectar repositorio de GitHub

1. Sube esta carpeta a un repositorio en GitHub.
2. Entra a <https://vercel.com/new> e importa el repo.
3. Vercel lee `vercel.json` y sirve `public/` automaticamente.

En ambos casos obtendras una URL como `https://titanic-predictor-xxxx.vercel.app`
que cualquiera puede abrir.

---

## Estructura del proyecto

```
titanic-predictor/
├── setup.py                    # Instala y parcha dependencias (corre 1 vez)
├── train_model.py              # Entrena la red neuronal con Keras
├── train_model.ipynb           # Alternativa: Colab si falla local
├── requirements.txt            # Dependencias (referencia, setup.py las maneja)
├── titanic-train.csv           # Dataset del profesor (915 pasajeros)
├── titanic-test.csv            # Dataset de evaluacion (393 pasajeros)
├── vercel.json                 # Configuracion de despliegue
├── .vercelignore               # Lo que NO sube a Vercel
├── INSTRUCCIONES.md            # Este archivo
│
├── mimodelo.json               # ← Generado (entregable Keras)
├── mimodelo.weights.h5         # ← Generado (entregable Keras)
│
└── public/                     # ← Lo unico que Vercel despliega
    ├── index.html              # UI completa (diseno Archivo Maritimo + TF.js)
    └── model/
        ├── model.json          # ← Generado (estructura TF.js)
        └── group1-shard1of1.bin # ← Generado (pesos en binario)
```

---

## Como las predicciones vienen del `.h5`

```
       (Tu maquina — train_model.py, una vez)
       ┌─────────────────────────────────────┐
       │ 1. Entrenar Keras Sequential        │
       │ 2. Guardar: mimodelo.weights.h5  ◄──┼── entregable
       │ 3. Recargar el .h5 y convertir a    │
       │    formato TF.js (mismos pesos)     │
       └─────────────────────────────────────┘
                          │
                          ▼
       (Browser, cada vez que alguien usa la pagina)
       ┌─────────────────────────────────────┐
       │ 4. tf.loadLayersModel('model.json') │
       │ 5. model.predict(input) ──► prob.   │
       └─────────────────────────────────────┘
```

El paso 3 NO reentrena: `tfjs.converters.save_keras_model()` lee los pesos
del modelo Keras que recien cargaste con `load_weights('mimodelo.weights.h5')`
y los re-empaqueta en archivos que el browser puede abrir.

---

## Detalles del modelo

**Inputs (orden importante, fijado en el entrenamiento):**

| Posicion | Variable | Tipo  | Notas                          |
|----------|----------|-------|--------------------------------|
| 0        | `Fare`   | float | Tarifa en libras esterlinas    |
| 1        | `Pclass` | int   | 1 = primera, 2 = segunda, 3 = tercera |
| 2        | `Gender` | int   | 0 = male, 1 = female           |
| 3        | `Age`    | float | Edad en años                   |
| 4        | `SibSp`  | int   | Hermanos + conyuge a bordo     |

**Arquitectura:** `Dense(32, ReLU) -> Dense(32, ReLU) -> Dense(1, sigmoid)`
**Optimizador:** Adam, **loss:** binary cross-entropy, **epochs:** 400.

---

## Troubleshooting

**`pip install tensorflow` falla con "Could not find a version"**
→ Estas usando Python 3.14. Instala Python 3.12 (paso 1).

**`setup.py` falla con "Could not find a version that satisfies tensorflow~=2.15.0"**
→ Tienes algun proceso que sigue intentando instalar TFDF. Verifica que estas
usando `py -3.12 setup.py` (no el Python por default).

**El browser muestra "Error: An InputLayer should be passed batchInputShape..."**
→ El modelo se entreno con Keras 3 (formato nuevo). `train_model.py` ya fuerza
Keras 2 con `TF_USE_LEGACY_KERAS=1`. Si lo cambiaste, restablece esa linea.

**`vercel deploy` sube los archivos .h5 o .csv**
→ Verifica que tienes `.vercelignore` en la raiz del proyecto.

---

## Tecnologias

- **Entrenamiento:** Python 3.12 + Keras 2 (`tf_keras`) sobre TensorFlow 2.21
- **Conversion a web:** `tensorflowjs.converters.save_keras_model()`
- **Frontend:** HTML + CSS + React 18 (CDN) + TensorFlow.js 4.22 (CDN)
- **Hosting:** Vercel (static, gratis, sin backend)

---
