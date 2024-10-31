#  Práctica 7. ML Governance (ML + OPS) 

## Objetivo de la práctica:

Al finalizar la práctica, serás capaz de:

- Comprender los principios fundamentales de ML Governance. <br>
- Implementar prácticas de gobernanza en las fases de desarrollo, entrega y operaciones de un proyecto de ML.<br>
- Aprender a gestionar el ciclo de vida completo de un modelo de ML.<br>
- Familiarizarse con herramientas y técnicas para asegurar la calidad, reproducibilidad y ética en proyectos de ML.

## Duración aproximada:

- 30 minutos.

## Problema a desarrollar:

**Sistema de Detección de Fraude en Transacciones Bancarias.**

Imagina que trabajas para un banco que está desarrollando un sistema de detección de fraude basado en machine learning. Tu tarea es implementar este sistema siguiendo las mejores prácticas de ML Governance a lo largo de todo el ciclo de vida del proyecto.

## Instrucciones:

### Fase 1: Development.

**Paso 1.** Definición del problema y planificación.

1. Crea un documento de especificación del proyecto (`project_spec.md`):

```markdown
 Proyecto de Detección de Fraude
 Objetivo
Desarrollar un sistema de ML para detectar transacciones fraudulentas en tiempo real.
 Métricas de Éxito
- Precision mínima del 95%
- Recall mínimo del 90%
- Tiempo de respuesta < 100ms por transacción
 Consideraciones Éticas
- Minimizar falsos positivos para evitar inconvenientes a clientes legítimos
- Asegurar la privacidad de los datos de los clientes
- Evitar sesgos basados en características protegidas (edad, género, etnia, etc.)
 Stakeholders
- Equipo de ML: Desarrollo y mantenimiento del modelo
- Equipo de Seguridad: Proporciona conocimiento del dominio y valida resultados
- Equipo Legal: Asegura el cumplimiento de regulaciones (GDPR, etc.)
- Equipo de TI: Responsable de la infraestructura y despliegue
```

**Paso 2.** Preparación y análisis de datos.

1. Crea un script para la preparación de datos (`src/data_preparation.py`):

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
def load_and_preprocess_data(file_path):
     Cargar datos
    data = pd.read_csv(file_path)
    
     Separar características y etiquetas
    X = data.drop('is_fraud', axis=1)
    y = data['is_fraud']
    
     Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
     Escalar características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
     Guardar el scaler para uso futuro
    joblib.dump(scaler, 'models/scaler.joblib')
    
    return X_train_scaled, X_test_scaled, y_train, y_test
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data('data/transactions.csv')
     Guardar los datos procesados
    np.save('data/processed/X_train.npy', X_train)
    np.save('data/processed/X_test.npy', X_test)
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/y_test.npy', y_test)
```
**Paso 3.** Desarrollo del modelo.

1. Crea un script para entrenar el modelo (`src/train_model.py`):

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
import joblib
import mlflow
def train_and_evaluate_model():
     Cargar datos procesados
    X_train = np.load('data/processed/X_train.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_test = np.load('data/processed/y_test.npy')
    
     Iniciar el seguimiento de MLflow
    mlflow.set_experiment("Fraud Detection Model")
    
    with mlflow.start_run():
         Entrenar el modelo
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
         Evaluar el modelo
        y_pred = model.predict(X_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
         Registrar métricas y parámetros
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
         Guardar el modelo
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
    
    return model
if __name__ == "__main__":
    model = train_and_evaluate_model()
    joblib.dump(model, 'models/fraud_detection_model.joblib')
```

### Fase 2. Delivery.

**Paso 4.** Pruebas y validación.

1. Crea un script para realizar pruebas adicionales (`src/model_testing.py`):

```python
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
def test_model():
     Cargar el modelo y los datos de prueba
    model = joblib.load('models/fraud_detection_model.joblib')
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    
     Realizar predicciones
    y_pred = model.predict(X_test)
    
     Generar informe de clasificación
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)
    
     Generar matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
if __name__ == "__main__":
    test_model()
```

**Paso 5.** Documentación.

5. Crea un documento de modelo (`model_card.md`):

```markdown
 Model Card: Fraud Detection System
 Model Details
- Developer: [Your Name]
- Model Date: [Current Date]
- Model Version: 1.0
- Model Type: Random Forest Classifier
 Intended Use
- Primary Use: Detect fraudulent transactions in real-time
- Intended Users: Bank's fraud detection team
 Training Data
- Source: Historical transaction data from [Date Range]
- Preprocessing: Standard scaling of numerical features
 Evaluation Data
- 20% hold-out test set from the original dataset
 Ethical Considerations
- The model has been tested for bias against protected characteristics
- Privacy measures are in place to protect customer data
 Caveats and Recommendations
- The model should be retrained periodically with new data
- Human oversight is recommended for final decision-making on flagged transactions
```
 Paso 6: Configuración del Pipeline de CI/CD
6. Crea un archivo de configuración para GitHub Actions (`.github/workflows/ci_cd.yml`):
```yaml
name: CI/CD Pipeline
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Run tests
      run: python -m pytest tests/
    - name: Train model
      run: python src/train_model.py
    - name: Run additional tests
      run: python src/model_testing.py
```

### Fase 3: Operations.

**Paso 7.** Despliegue del modelo.

1. Crea un script para servir el modelo (`src/serve_model.py`):

```python
from flask import Flask, request, jsonify
import joblib
import numpy as np
app = Flask(__name__)
 Cargar el modelo y el scaler
model = joblib.load('models/fraud_detection_model.joblib')
scaler = joblib.load('models/scaler.joblib')
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(list(data.values())).reshape(1, -1)
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    return jsonify({'fraud_prediction': int(prediction)})
if __name__ == '__main__':
    app.run(debug=True)
```

**Paso 8.** Monitoreo y logging.

8. Agrega logging y monitoreo al script de servicio (`src/serve_model.py`):

```python
from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging
from prometheus_client import Counter, Histogram
from flask_prometheus_metrics import register_metrics
app = Flask(__name__)
 Configurar logging
logging.basicConfig(filename='fraud_detection.log', level=logging.INFO)
 Métricas de Prometheus
PREDICTIONS = Counter('fraud_predictions_total', 'Total number of predictions')
RESPONSE_TIME = Histogram('prediction_response_time_seconds', 'Response time for predictions')
 Registrar métricas
register_metrics(app)
 Cargar el modelo y el scaler
model = joblib.load('models/fraud_detection_model.joblib')
scaler = joblib.load('models/scaler.joblib')
@app.route('/predict', methods=['POST'])
@RESPONSE_TIME.time()
def predict():
    data = request.json
    features = np.array(list(data.values())).reshape(1, -1)
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    PREDICTIONS.inc()
    logging.info(f"Prediction made: {prediction}")
    return jsonify({'fraud_prediction': int(prediction)})
if __name__ == '__main__':
    app.run(debug=True)
```

**Paso 9.** Mantenimiento y actualización.

1. Crea un script para reentrenar el modelo periódicamente (`src/retrain_model.py`):

```python
import schedule
import time
from train_model import train_and_evaluate_model
def retrain_job():
    print("Retraining model...")
    train_and_evaluate_model()
    print("Model retrained and saved.")
 Programar el reentrenamiento para que ocurra cada semana
schedule.every().monday.at("02:00").do(retrain_job)
if __name__ == "__main__":
    while True:
        schedule.run_pending()
        time.sleep(1)
```
 
## Resultado esperado:

Este laboratorio te ha introducido a los conceptos clave de ML Governance a lo largo del ciclo de vida completo de un proyecto de ML:

1. **Development**: <br>
   - Definición clara del problema y planificación.<br>
   - Preparación y análisis de datos éticos.<br>
   - Desarrollo del modelo con seguimiento de experimentos (MLflow).

2. **Delivery**:<br>
   - Pruebas exhaustivas y validación.<br>
   - Documentación detallada (Model Card).<br>
   - Implementación de un pipeline de CI/CD.

3. **Operations**:<br>
   - Despliegue del modelo como un servicio web.<br>
   - Monitoreo y logging para seguimiento en producción.<br>
   - Mantenimiento y actualización periódica del modelo.

Para mejorar este proyecto, podrías:

- Implementar pruebas de equidad y sesgo más exhaustivas.<br>
- Agregar explicabilidad al modelo (por ejemplo, usando SHAP values).<br>
- Implementar un sistema de versionado de datos y modelos más robusto.<br>
- Configurar alertas basadas en el rendimiento del modelo en producción.<br>
- Implementar un sistema de feedback loop para mejorar continuamente el modelo con nuevos datos.

### [Índice](../README.md)

### [Práctica 6. Proceso de despliegue de un modelo de Machine Learning](../Capítulo6/README.md)

### [Práctica 8. Aplicación de Machine Learning en el sector aduanero](../Capítulo8/README.md)
