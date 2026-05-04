from pathlib import Path


THRESHOLD = 3000
METRIC_PATH = Path("metrics/mse.txt")


if not METRIC_PATH.exists():
    raise FileNotFoundError(
        "No se encontró metrics/mse.txt. Ejecuta primero src/train.py."
    )


mse = float(METRIC_PATH.read_text(encoding="utf-8"))

print(f"MSE obtenido: {mse:.4f}")
print(f"Umbral definido: {THRESHOLD}")

if mse < THRESHOLD:
    print("Validación exitosa: el modelo cumple el umbral de calidad.")
else:
    raise ValueError(
        f"Validación fallida: el MSE ({mse:.4f}) es mayor o igual al umbral ({THRESHOLD})."
    )