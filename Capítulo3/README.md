# Práctica 3. Mejora del proceso de aprobación de créditos 

## Objetivo de la práctica:

Al finalizar la práctica, serás capaz de:

- Implementar trackeo de experimentos. <br>
- Realizar versionamiento de código y modelos.<br>
- Calibrar el modelo seleccionado.<br>
- Detectar y manejar concept drift.

## Duración aproximada:

- 60 minutos.

## Problema a desarrollar:

Una empresa de préstamos quiere mejorar su proceso de aprobación de créditos utilizando machine learning. 

Tu tarea es desarrollar un modelo de clasificación para predecir si un solicitante pagará el préstamo a tiempo o no. Deberás experimentar con diferentes modelos, realizar un seguimiento de tus experimentos, versionar tu código y modelo, calibrar el modelo final y monitorear el rendimiento del modelo a lo largo del tiempo para detectar concept drift.

## Instrucciones:

### Tarea 1. Preparación del entorno y datos.

**Paso 1.** Primero, vamos a instalar las bibliotecas necesarias y preparar nuestros datos.

```
bash
pip install pandas numpy scikit-learn mlflow joblib
```

**Paso 2.** Ahora, crea un archivo `prepare_data.py`:

```
python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
 Generar datos sintéticos
np.random.seed(42)
n_samples = 1000
data = pd.DataFrame({
    'income': np.random.normal(50000, 15000, n_samples),
    'age': np.random.randint(18, 70, n_samples),
    'employment_length': np.random.randint(0, 20, n_samples),
    'loan_amount': np.random.normal(10000, 5000, n_samples),
    'credit_score': np.random.normal(700, 100, n_samples)
})
 Crear variable objetivo (0: no paga a tiempo, 1: paga a tiempo)
data['target'] = (0.4 * data['income'] / 50000 +
                  0.2 * data['age'] / 50 +
                  0.1 * data['employment_length'] / 10 +
                  0.2 * data['credit_score'] / 700 -
                  0.1 * data['loan_amount'] / 10000 +
                  np.random.normal(0, 0.1, n_samples)) > 0.5
data['target'] = data['target'].astype(int)
 Dividir en conjuntos de entrenamiento y prueba
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 Escalar características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
 Guardar datos
np.save('X_train.npy', X_train_scaled)
np.save('X_test.npy', X_test_scaled)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
print("Datos preparados y guardados.")
```

**Paso 3.** Ejecuta el script para preparar los datos:

```bash
python prepare_data.py
```

### Tarea 2. Trackeo de Experimentos con MLflow.

Vamos a usar MLflow para realizar un seguimiento de nuestros experimentos. 

**Paso 1.** Crea un archivo `train_model.py`:

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import mlflow
import mlflow.sklearn
def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return accuracy, roc_auc
 Cargar datos
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')
 Configurar MLflow
mlflow.set_experiment("loan_approval_prediction")
 Experimentar con diferentes modelos
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}
for name, model in models.items():
    with mlflow.start_run(run_name=name):
        accuracy, roc_auc = train_and_evaluate(model, X_train, y_train, X_test, y_test)
        
         Registrar parámetros y métricas
        mlflow.log_params(model.get_params())
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", roc_auc)
        
         Guardar el modelo
        mlflow.sklearn.log_model(model, "model")
        
        print(f"{name} - Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")
print("Experimentos completados y registrados en MLflow.")
```

**Paso 2.** Ejecuta el script para entrenar los modelos y realiza el seguimiento de experimentos:

```bash
python train_model.py
```

### Tarea 3. Versionamiento con Git y MLflow.

**Paso 1.** Inicializa un repositorio Git y realiza el primer commit.

```bash
git init
git add prepare_data.py train_model.py
git commit -m "Initial commit: data preparation and model training scripts"
```

MLflow ya está versionando nuestros modelos automáticamente. Puedes ver los experimentos ejecutando:

```bash
mlflow ui
```

**Paso 2.** Visita `http://localhost:5000` en tu navegador para ver los resultados de los experimentos.

### Tarea 4. Calibración de modelo.

Vamos a calibrar el mejor modelo (asumamos que es el Random Forest). 

**Paso 1.** Crea un archivo `calibrate_model.py`:

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
import mlflow
import mlflow.sklearn
 Cargar datos
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')
 Cargar el mejor modelo (Random Forest)
best_run = mlflow.search_runs(experiment_names=["loan_approval_prediction"]).iloc[0]
model_uri = f"runs:/{best_run.run_id}/model"
best_model = mlflow.sklearn.load_model(model_uri)
 Calibrar el modelo
calibrated_model = CalibratedClassifierCV(best_model, cv=5, method='sigmoid')
calibrated_model.fit(X_train, y_train)
 Evaluar el modelo calibrado
y_prob_uncalibrated = best_model.predict_proba(X_test)[:, 1]
y_prob_calibrated = calibrated_model.predict_proba(X_test)[:, 1]
brier_uncalibrated = brier_score_loss(y_test, y_prob_uncalibrated)
brier_calibrated = brier_score_loss(y_test, y_prob_calibrated)
print(f"Brier score (uncalibrated): {brier_uncalibrated:.4f}")
print(f"Brier score (calibrated): {brier_calibrated:.4f}")
 Guardar el modelo calibrado con MLflow
with mlflow.start_run(run_name="Calibrated Random Forest"):
    mlflow.log_metric("brier_score", brier_calibrated)
    mlflow.sklearn.log_model(calibrated_model, "calibrated_model")
print("Modelo calibrado y guardado en MLflow.")
```

**Paso 2.** Ejecuta el script para calibrar el modelo:

```bash
python calibrate_model.py
```

### Tarea 5. Detección de Concept Drift.
 
Para simular y detectar concept drift, vamos a crear un script que genere nuevos datos con una distribución ligeramente diferente y compare el rendimiento del modelo. 

**Paso 1.** Crea un archivo `detect_drift.py`:

```python
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
def generate_drift_data(n_samples=1000, drift_factor=0.1):
    np.random.seed(42)
    data = pd.DataFrame({
        'income': np.random.normal(50000 * (1 + drift_factor), 15000, n_samples),
        'age': np.random.randint(18, 70, n_samples),
        'employment_length': np.random.randint(0, 20, n_samples),
        'loan_amount': np.random.normal(10000, 5000, n_samples),
        'credit_score': np.random.normal(700 * (1 - drift_factor), 100, n_samples)
    })
    
    data['target'] = (0.4 * data['income'] / 50000 +
                      0.2 * data['age'] / 50 +
                      0.1 * data['employment_length'] / 10 +
                      0.2 * data['credit_score'] / 700 -
                      0.1 * data['loan_amount'] / 10000 +
                      np.random.normal(0, 0.1, n_samples)) > 0.5
    data['target'] = data['target'].astype(int)
    
    return data
 Cargar el modelo calibrado
best_run = mlflow.search_runs(experiment_names=["loan_approval_prediction"]).iloc[0]
model_uri = f"runs:/{best_run.run_id}/calibrated_model"
model = mlflow.sklearn.load_model(model_uri)
 Cargar datos originales de prueba
X_test_original = np.load('X_test.npy')
y_test_original = np.load('y_test.npy')
 Generar nuevos datos con drift
drift_data = generate_drift_data()
X_drift = drift_data.drop('target', axis=1)
y_drift = drift_data['target']
 Evaluar el modelo en datos originales y nuevos
accuracy_original = accuracy_score(y_test_original, model.predict(X_test_original))
accuracy_drift = accuracy_score(y_drift, model.predict(X_drift))
print(f"Accuracy en datos originales: {accuracy_original:.4f}")
print(f"Accuracy en datos con drift: {accuracy_drift:.4f}")
 Detectar drift
drift_threshold = 0.05
if abs(accuracy_original - accuracy_drift) > drift_threshold:
    print("¡Alerta! Se ha detectado concept drift.")
    print("Se recomienda reentrenar el modelo con datos más recientes.")
else:
    print("No se ha detectado concept drift significativo.")
 Registrar resultados en MLflow
with mlflow.start_run(run_name="Concept Drift Detection"):
    mlflow.log_metric("accuracy_original", accuracy_original)
    mlflow.log_metric("accuracy_drift", accuracy_drift)
    mlflow.log_metric("accuracy_difference", abs(accuracy_original - accuracy_drift))
print("Resultados de detección de drift registrados en MLflow.")
```

**Paso 2.** Ejecuta el siguiente script para detectar concept drift:

```bash
python detect_drift.py
```

## Resultado esperado:

En este laboratorio, has aprendido a:

1. Preparar datos para un problema de aprobación de préstamos.<br>
2. Utilizar MLflow para el trackeo de experimentos con diferentes modelos.<br>
3. Versionar tu código con Git y tus modelos con MLflow.<br>
4. Calibrar el mejor modelo para mejorar sus predicciones de probabilidad.<br>
5. Detectar concept drift simulando cambios en la distribución de los datos.<br>

Estos pasos son fundamentales en el ciclo de vida de un proyecto de machine learning, asegurando que puedas desarrollar, evaluar y mantener modelos de manera efectiva.

Para continuar mejorando este proceso, podrías considerar:

- Implementar técnicas de selección de características. <br>
- Explorar más modelos y técnicas de optimización de hiperparámetros.<br>
- Desarrollar un pipeline de reentrenamiento automático cuando se detecte concept drift.<br>
- Implementar un sistema de monitoreo continuo del rendimiento del modelo en producción.<br>
