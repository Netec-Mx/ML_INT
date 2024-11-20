#  Práctica 7. ML Governance (ML + OPS) 

## Objetivo de la práctica:

Al finalizar la práctica, serás capaz de:

- Comprender los principios fundamentales de ML Governance. <br>
- Implementar prácticas de gobernanza en las fases de desarrollo, entrega y operaciones de un proyecto de ML.<br>
- Aprender a gestionar el ciclo de vida completo de un modelo de ML.<br>
- Familiarizarse con herramientas y técnicas para asegurar la calidad, reproducibilidad y ética en proyectos de ML.

## Duración aproximada:

- 30 minutos.

## Prerrequisitos

- Debes tener una cuenta activa de GITHUB o en su defecto crear una, **antes de iniciar la practica**
- Crear un repositorio privado llamado **intmllab7**
- **Clonar** el repositorio en tu ambiente de Visual Studio Code
  
## Problema a desarrollar:

**Sistema de Detección de Fraude en Transacciones Bancarias.**

Imagina que trabajas para un banco que está desarrollando un sistema de detección de fraude basado en machine learning. Tu tarea es implementar este sistema siguiendo las mejores prácticas de ML Governance a lo largo de todo el ciclo de vida del proyecto.

## Instrucciones:

### Fase 1: Development.

**Paso 0.** En raíz del directorio en que vayas a trabjar crea el archivo `requirements.txt` con las siguientes librerías necesarias.

```
# Librerías principales
pandas
numpy
scikit-learn
joblib

# MLflow para seguimiento de experimentos
mlflow

# Flask para servir el modelo como API
Flask

# Prometheus para monitoreo
prometheus-client
flask-prometheus-metrics

# Schedule para tareas periódicas
schedule

# Pytest para pruebas
pytest
pytest-flask
```

- Crear el archivo `data/transactions.csv`:

```
is_fraud,amount,time_since_last_transaction,risk_score
0,100.50,24,0.1
1,7500.00,2,0.9
0,200.30,48,0.2
1,15000.00,1,0.95
0,320.00,36,0.15
1,10000.00,0.5,0.85
0,180.25,60,0.05
1,5000.50,0.3,0.9
0,250.75,72,0.12
1,20000.00,0.1,0.99
``` 

**Paso 1.** Definición del problema y planificación.

1. Crea un documento de especificación del proyecto (`project_spec.md`):

```
# Proyecto de Detección de Fraude

## Objetivo
Desarrollar un sistema de ML para detectar transacciones fraudulentas en tiempo real.

## Métricas de Éxito
- Precision mínima del 95%
- Recall mínimo del 90%
- Tiempo de respuesta < 100ms por transacción

## Consideraciones Éticas
- Minimizar falsos positivos para evitar inconvenientes a clientes legítimos
- Asegurar la privacidad de los datos de los clientes
- Evitar sesgos basados en características protegidas (edad, género, etnia, etc.)

## Stakeholders
- Equipo de ML: Desarrollo y mantenimiento del modelo
- Equipo de Seguridad: Proporciona conocimiento del dominio y valida resultados
- Equipo Legal: Asegura el cumplimiento de regulaciones (GDPR, etc.)
- Equipo de TI: Responsable de la infraestructura y despliegue
```

**Paso 2.** Preparación y análisis de datos.

1. Crea un script para la preparación de datos (`src/data_preparation.py`):

```
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_and_preprocess_data(file_path):
    print("Cargando y procesando datos...")
    
    # Cargar datos
    data = pd.read_csv(file_path)
    
    # Separar características y etiquetas
    X = data[['amount', 'time_since_last_transaction', 'risk_score']]
    y = data['is_fraud']
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Escalar características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Crear directorios si no existen
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Guardar datos y scaler
    np.save('data/processed/X_train.npy', X_train_scaled)
    np.save('data/processed/X_test.npy', X_test_scaled)
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/y_test.npy', y_test)
    joblib.dump(scaler, 'models/scaler.joblib')
    
    print("Datos procesados guardados correctamente.")
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    load_and_preprocess_data('data/transactions.csv')
```

**Paso 3.** Pruebas Unitarias para Preparación de Datos.

1. Crea el archivo (`tests/test_data_preparation.py`):

```
import pytest
import os  # Asegurarse de importar el módulo os
from src.data_preparation import load_and_preprocess_data

def test_load_and_preprocess_data():
    # Usar el archivo de datos correcto
    file_path = 'data/transactions.csv'
    assert os.path.exists(file_path), f"El archivo {file_path} no existe"
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)
    assert X_train.shape[0] > 0, "X_train debería contener datos"
    assert y_train.shape[0] > 0, "y_train debería contener etiquetas"
```

**Paso 4.** Desarrollo del modelo.

1. Crea un script para entrenar el modelo (`src/train_model.py`):

```
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
import joblib
import mlflow

def train_and_evaluate_model():
    print("Iniciando entrenamiento del modelo...")
    
    # Cargar datos procesados
    X_train = np.load('data/processed/X_train.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    # Iniciar el seguimiento de MLflow
    mlflow.set_experiment("Fraud Detection Model")
    
    with mlflow.start_run():
        # Entrenar el modelo
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluar el modelo
        y_pred = model.predict(X_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        # Registrar métricas y guardar modelo
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.sklearn.log_model(model, "model", input_example=X_train[:1])
        
        # Guardar el modelo
        model_path = 'models/fraud_detection_model.joblib'
        joblib.dump(model, model_path)
        print(f"Modelo guardado en: {model_path}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        return model

if __name__ == "__main__":
    train_and_evaluate_model()
```

**Paso 5.** Pruebas para Entrenamiento de Modelo.

1. Crea el archivo (`tests/test_train_model.py`):

```
import os
from src.train_model import train_and_evaluate_model

def test_train_and_evaluate_model():
    model = train_and_evaluate_model()
    assert os.path.exists('models/fraud_detection_model.joblib'), "El modelo debería guardarse correctamente"
    assert model is not None, "El modelo debería entrenarse correctamente"
```

### Fase 2. Delivery.

**Paso 6.** Validación del modelo.

1. Crea un script (`src/model_testing.py`):

```
import joblib
import numpy as np
from sklearn.metrics import classification_report

def test_model():
    model = joblib.load('models/fraud_detection_model.joblib')
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)
```

**Paso 7.** Pruebas de Integración

1. Crea un script (`tests/test_model_testing.py`):

```
import pytest
import joblib
import numpy as np
from sklearn.metrics import classification_report

def test_model():
    # Verificar que el modelo existe
    model_path = 'models/fraud_detection_model.joblib'
    assert joblib.load(model_path), f"El modelo no está disponible en {model_path}"
    
    # Cargar datos de prueba
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    # Generar predicciones
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    
    # Generar y validar informe de clasificación
    report = classification_report(y_test, y_pred, output_dict=True)
    assert 'accuracy' in report, "El informe debería contener 'accuracy'"
```

**Paso 8.** Documentación.

5. Crea un documento de modelo (`src/model_card.md`):

```
# Model Card: Fraud Detection System

## Model Details
- Developer: [Your Name]
- Model Date: [Current Date]
- Model Version: 1.0
- Model Type: Random Forest Classifier
 
## Intended Use
- Primary Use: Detect fraudulent transactions in real-time
- Intended Users: Bank's fraud detection team

## Training Data
- Source: Historical transaction data from [Date Range]
- Preprocessing: Standard scaling of numerical features

## Evaluation Data
- 20% hold-out test set from the original dataset

## Ethical Considerations
- The model has been tested for bias against protected characteristics
- Privacy measures are in place to protect customer data

## Caveats and Recommendations
- The model should be retrained periodically with new data
- Human oversight is recommended for final decision-making on flagged transactions
```

**Paso 9:** Configuración del Pipeline de CI/CD

6. Crea un archivo de configuración para GitHub Actions (`.github/workflows/ci_cd.yml`):

```
name: CI/CD Pipeline
on:
  push:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.8'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt

    - name: Prepare data
      run: python src/data_preparation.py  # Generar datos procesados

    - name: Train model
      run: python src/train_model.py  # Entrenar el modelo

    - name: Verify model file
      run: ls models/fraud_detection_model.joblib || echo "Model file missing!"

    - name: Run tests
      run: python -m pytest tests/
```

### Fase 3: Operations.

**Paso 10.** Despliegue y monitoreo.

1. Crea un script para servir el modelo (`src/serve_model.py`):

```
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('models/fraud_detection_model.joblib')
scaler = joblib.load('models/scaler.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(list(data.values())).reshape(1, -1)
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    return jsonify({'fraud_prediction': int(prediction)})
```

**Paso 11.** Pruebas del Servicio del Modelo

1. Crea el archivo (`tests/test_model_serving.py`):

```
import os
import pytest
from src.serve_model import app

@pytest.fixture(scope="module")
def client():
    # Verificar si el modelo existe antes de las pruebas
    assert os.path.exists('models/fraud_detection_model.joblib'), "El modelo no está disponible en 'models/fraud_detection_model.joblib'."
    return app.test_client()

def test_predict_endpoint(client):
    # Crear datos de prueba
    test_data = {"feature1": 0.5, "feature2": -1.0, "feature3": 2.1}
    
    # Hacer una solicitud POST al endpoint /predict
    response = client.post('/predict', json=test_data)
    
    # Verificar respuesta
    assert response.status_code == 200, "El endpoint debería devolver un código 200."
    json_data = response.get_json()
    assert 'fraud_prediction' in json_data, "La respuesta debería contener 'fraud_prediction'."
```

**Paso 12.** Mantenimiento y actualización.

1. Crea un script para reentrenar el modelo periódicamente (`src/retrain_model.py`):

```
import schedule
import time
from src.train_model import train_and_evaluate_model

def retrain_job():
    print("Retraining model...")
    train_and_evaluate_model()
    print("Model retrained and saved.")
    
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
