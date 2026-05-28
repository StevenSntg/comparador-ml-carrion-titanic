"""
Instalacion automatica de dependencias para train_model.py.

Resuelve dos problemas conocidos al combinar TF 2.21 + tensorflowjs 4.22 + Python 3.12:

  1. tensorflowjs depende de `tensorflow_decision_forests` que pinea TF a 2.15
     (incompatible con TF 2.16+ que necesitamos para Python 3.12).
     -> Instalamos tensorflowjs con --no-deps y parchamos el import para que
        sea opcional. No necesitamos TFDF para modelos Sequential.

  2. tensorflowjs importa `jax` indirectamente, no requerido en Windows.
     -> Parchamos ese import tambien.

  3. tensorflow_hub usa `pkg_resources` que fue removido en setuptools 81+.
     -> Pineamos setuptools a <81.

REQUISITO: Python 3.10-3.13 (no 3.14, TF aun no soporta).

USO:
    py -3.12 setup.py
"""
import os
import sys
import subprocess


def run(cmd):
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\n[ERROR] El comando fallo (codigo {result.returncode})")
        sys.exit(result.returncode)


def patch_file(path, old_text, new_text, marker):
    """Aplica un parche idempotente a un archivo."""
    if not os.path.exists(path):
        print(f"  [!] No se encontro {path} -- saltando parche")
        return
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if marker in content:
        print(f"  Ya parchado: {os.path.basename(path)}")
        return
    if old_text not in content:
        print(f"  [!] No se encontro el patron en {path} -- saltando")
        return
    content = content.replace(old_text, new_text)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  OK: {os.path.basename(path)}")


def main():
    print("=" * 70)
    print(" Setup de dependencias para train_model.py")
    print("=" * 70)
    print(f"\nPython: {sys.version}")

    if sys.version_info >= (3, 14):
        print("\n[ERROR] Python 3.14 no es compatible con TensorFlow.")
        print("Usa: py -3.12 setup.py")
        sys.exit(1)

    if sys.version_info < (3, 10):
        print("\n[ERROR] TensorFlow requiere Python 3.10 o superior.")
        sys.exit(1)

    # ─── 1. Dependencias principales ─────────────────────────────────────────
    print("\n[1/4] Instalando tensorflow, tf-keras, pandas, numpy ...")
    run([sys.executable, "-m", "pip", "install",
         "tensorflow>=2.16",
         "tensorflow-hub",
         "tf-keras",
         "pandas>=2.0",
         "numpy>=1.26,<2.0"])

    # ─── 2. setuptools < 81 (para pkg_resources) ─────────────────────────────
    print("\n[2/4] Forzando setuptools<81 (necesario para pkg_resources) ...")
    run([sys.executable, "-m", "pip", "install", "setuptools<81"])

    # ─── 3. tensorflowjs sin sus deps conflictivas ───────────────────────────
    print("\n[3/4] Instalando tensorflowjs (sin deps conflictivas) ...")
    run([sys.executable, "-m", "pip", "install", "--no-deps", "tensorflowjs>=4.22"])

    # ─── 4. Parches a tensorflowjs ───────────────────────────────────────────
    print("\n[4/4] Aplicando parches a tensorflowjs ...")
    # Localizamos tensorflowjs sin importarlo (el import falla antes del parche).
    import importlib.util
    spec = importlib.util.find_spec("tensorflowjs")
    if spec is None or spec.origin is None:
        print("[ERROR] No se encontro tensorflowjs en site-packages")
        sys.exit(1)
    tfjs_root = os.path.dirname(spec.origin)

    # Parche A: convertirs el import de tensorflow_decision_forests en opcional
    patch_file(
        os.path.join(tfjs_root, "converters", "tf_saved_model_conversion_v2.py"),
        old_text="# Required to load saved models that use TFDF.\nimport tensorflow_decision_forests",
        new_text=(
            "# Required to load saved models that use TFDF.\n"
            "# Patched: TFDF no esta disponible para Python 3.12 + TF 2.21 en Windows.\n"
            "try:\n"
            "    import tensorflow_decision_forests\n"
            "except ImportError:\n"
            "    tensorflow_decision_forests = None"
        ),
        marker="# Patched: TFDF no esta disponible",
    )

    # Parche B: convertir el import de jax en opcional
    patch_file(
        os.path.join(tfjs_root, "converters", "__init__.py"),
        old_text="from tensorflowjs.converters.jax_conversion import convert_jax",
        new_text=(
            "# Patched: jax es opcional, solo para conversion de modelos JAX\n"
            "try:\n"
            "    from tensorflowjs.converters.jax_conversion import convert_jax\n"
            "except ImportError:\n"
            "    convert_jax = None"
        ),
        marker="# Patched: jax es opcional",
    )

    # ─── 5. Verificacion ────────────────────────────────────────────────────
    print("\nVerificando que todo se instalo correctamente ...")
    test = subprocess.run(
        [sys.executable, "-c",
         "import os; os.environ['TF_USE_LEGACY_KERAS']='1'; "
         "import tf_keras; import tensorflowjs; "
         "print(f'tf_keras: {tf_keras.__version__}'); "
         "print(f'tensorflowjs: {tensorflowjs.__version__}')"],
        capture_output=True, text=True,
    )
    if test.returncode == 0:
        print(test.stdout)
        print("=" * 70)
        print(" SETUP COMPLETADO")
        print("=" * 70)
        print("\nAhora puedes entrenar el modelo:")
        print(f"  {os.path.basename(sys.executable)} train_model.py")
    else:
        print("[ERROR] Algo fallo en la verificacion final:")
        print(test.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
