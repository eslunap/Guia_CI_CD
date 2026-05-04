import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# Crear carpetas necesarias
os.makedirs("models", exist_ok=True)
os.makedirs("metrics", exist_ok=True)

# Configurar MLflow con tracking local
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("diabetes-regression-ci-cd")

# Cargar datos
data = load_diabetes()
X = data.data
y = data.target

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Entrenar modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluar modelo
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Guardar modelo localmente
model_path = "models/linear_regression_model.pkl"
joblib.dump(model, model_path)

# Guardar métrica localmente
with open("metrics/mse.txt", "w", encoding="utf-8") as f:
    f.write(str(mse))

# Registrar en MLflow
with mlflow.start_run():
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("dataset", "sklearn_diabetes")
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_artifact(model_path)
    mlflow.log_artifact("metrics/mse.txt")

print(f"Modelo entrenado correctamente. MSE: {mse:.4f}")