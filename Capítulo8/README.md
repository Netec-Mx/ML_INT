# Práctica 8. Aplicación de Machine Learning en el sector aduanero 

## Objetivo de la práctica:

Al finalizar la práctica, serás capaz de:

- Comprender cómo aplicar el machine learning en el sector aduanero.
- Desarrollar un modelo de ML para un caso de uso específico en aduanas.
- Aprender a manejar datos típicos del sector aduanero.
- Implementar un sistema de ML completo, desde la preparación de datos hasta el despliegue y monitoreo.

## Duración aproximada:

- 25 minutos.

## Problema a desarrollar:

**Sistema de Clasificación Arancelaria Automatizada.**

Imagina que trabajas para la Aduana Nacional y te han encargado desarrollar un sistema de clasificación arancelaria automatizada utilizando machine learning. Este sistema ayudará a los agentes aduaneros a clasificar correctamente los productos importados según el Sistema Armonizado (SA) de designación y codificación de mercancías.

## Instrucciones:

### Tarea 1. Comprensión del problema y recopilación de datos.

**Paso 1.** Investiga sobre el Sistema Armonizado (SA) y la clasificación arancelaria.

**Paso 2.** Recopila un conjunto de datos de productos importados con sus descripciones y códigos SA correspondientes.

**Paso 3.** Crea un archivo CSV (`data/import_data.csv`) con la siguiente estructura:

```csv
id,description,hs_code
1,"Laptop computer, 15-inch screen, 8GB RAM, 256GB SSD",8471.30
2,"Men's cotton t-shirt, short sleeve, blue",6109.10
3,"Smartphone, 6.1-inch display, 128GB storage",8517.13
...
```
### Tarea 2. Preparación y análisis de datos.

**Paso 1.** Crea un script (`src/data_preparation.py`) para preparar los datos:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
def prepare_data(file_path):
     Cargar datos
    data = pd.read_csv(file_path)
    
     Dividir en características (X) y etiquetas (y)
    X = data['description']
    y = data['hs_code']
    
     Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
     Vectorizar las descripciones usando TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
     Guardar el vectorizador para uso futuro
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.joblib')
    
    return X_train_vectorized, X_test_vectorized, y_train, y_test
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data('data/import_data.csv')
     Guardar los datos procesados
    joblib.dump(X_train, 'data/processed/X_train.joblib')
    joblib.dump(X_test, 'data/processed/X_test.joblib')
    joblib.dump(y_train, 'data/processed/y_train.joblib')
    joblib.dump(y_test, 'data/processed/y_test.joblib')
```

### Tarea 3. Desarrollo del modelo.

**Paso 1.** Crea un script (`src/train_model.py`) para entrenar el modelo:

```python
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import mlflow
def train_and_evaluate_model():
     Cargar datos procesados
    X_train = joblib.load('data/processed/X_train.joblib')
    X_test = joblib.load('data/processed/X_test.joblib')
    y_train = joblib.load('data/processed/y_train.joblib')
    y_test = joblib.load('data/processed/y_test.joblib')
    
     Iniciar el seguimiento de MLflow
    mlflow.set_experiment("HS Code Classification")
    
    with mlflow.start_run():
         Entrenar el modelo
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
         Evaluar el modelo
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
         Registrar métricas y parámetros
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", report['accuracy'])
        mlflow.log_metric("weighted_avg_f1-score", report['weighted avg']['f1-score'])
        
         Guardar el modelo
        mlflow.sklearn.log_model(model, "model")
        
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
    
    return model
if __name__ == "__main__":
    model = train_and_evaluate_model()
    joblib.dump(model, 'models/hs_code_classifier.joblib')
```

### Tarea 4. Implementación del servicio de predicción.

**Paso 1.** Crea un script (`src/predict_service.py`) para implementar el servicio de predicción:

```python
from flask import Flask, request, jsonify
import joblib
import numpy as np
app = Flask(__name__)
 Cargar el modelo y el vectorizador
model = joblib.load('models/hs_code_classifier.joblib')
vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    description = data['description']
    
     Vectorizar la descripción
    vectorized_description = vectorizer.transform([description])
    
     Realizar la predicción
    prediction = model.predict(vectorized_description)[0]
    
    return jsonify({'hs_code': prediction})
if __name__ == '__main__':
    app.run(debug=True)
```

### Tarea 5. Pruebas y validación.

**Paso 1.** Crea un script (`src/test_model.py`) para realizar pruebas adicionales:

```python
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def test_model():
     Cargar el modelo y los datos de prueba
    model = joblib.load('models/hs_code_classifier.joblib')
    X_test = joblib.load('data/processed/X_test.joblib')
    y_test = joblib.load('data/processed/y_test.joblib')
    
     Realizar predicciones
    y_pred = model.predict(X_test)
    
     Generar informe de clasificación
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)
    
     Generar matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('reports/confusion_matrix.png')
    plt.close()
if __name__ == "__main__":
    test_model()
```

### Tarea 6. Documentación.

**Paso 1.** Crea un documento de modelo (`model_card.md`):

```markdown
 Model Card: HS Code Classifier
 Model Details
- Developer: [Your Name]
- Model Date: [Current Date]
- Model Version: 1.0
- Model Type: Random Forest Classifier
 Intended Use
- Primary Use: Assist customs officers in classifying imported goods according to the Harmonized System (HS)
- Intended Users: Customs officers and import/export specialists
 Training Data
- Source: Historical import data with product descriptions and corresponding HS codes
- Size: [Number of samples] records
- Preprocessing: TF-IDF vectorization of product descriptions
 Evaluation Data
- 20% hold-out test set from the original dataset
 Ethical Considerations
- The model should be used as a tool to assist human decision-making, not to replace it entirely
- Regular audits should be performed to ensure the model is not introducing or amplifying biases
 Caveats and Recommendations
- The model's performance may vary for product categories that are underrepresented in the training data
- The model should be retrained periodically with new data to stay up-to-date with changes in product descriptions and HS codes
- Users should be trained on how to interpret the model's predictions and when to seek additional verification
```
 
### Tarea 7. Despliegue y monitoreo.
**Paso 1.** Utiliza Docker para containerizar la aplicación. Crea un `Dockerfile`:

```dockerfile
FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ .
COPY models/ models/
CMD ["python", "predict_service.py"]
```
**Paso 2.** Crea un script (`src/monitor_service.py`) para monitorear el servicio:

```python
import requests
import time
import logging
from prometheus_client import start_http_server, Counter, Histogram
 Configurar logging
logging.basicConfig(filename='hs_code_classifier.log', level=logging.INFO)
 Métricas de Prometheus
PREDICTIONS = Counter('hs_code_predictions_total', 'Total number of predictions')
RESPONSE_TIME = Histogram('prediction_response_time_seconds', 'Response time for predictions')
def monitor_prediction_service():
    while True:
        try:
            start_time = time.time()
            response = requests.post('http://localhost:5000/predict', 
                                     json={'description': 'Sample product description'})
            duration = time.time() - start_time
            
            if response.status_code == 200:
                PREDICTIONS.inc()
                RESPONSE_TIME.observe(duration)
                logging.info(f"Prediction made: {response.json()}")
            else:
                logging.error(f"Error in prediction: {response.status_code}")
        
        except Exception as e:
            logging.error(f"Error in monitoring: {str(e)}")
        
        time.sleep(60)   Esperar 1 minuto antes de la próxima verificación
if __name__ == '__main__':
    start_http_server(8000)   Iniciar servidor de métricas de Prometheus
    monitor_prediction_service()
```

**Paso 3.** Configura Prometheus y Grafana para visualizar las métricas del servicio.

### Tarea 8. Mantenimiento y actualización.

**Paso 1.** Crea un script (`src/update_model.py`) para actualizar periódicamente el modelo:

```python
import schedule
import time
from train_model import train_and_evaluate_model
def update_model_job():
    print("Updating HS Code Classifier model...")
    train_and_evaluate_model()
    print("Model updated and saved.")
 Programar la actualización del modelo para que ocurra cada mes
schedule.every().month.at("02:00").do(update_model_job)
if __name__ == "__main__":
    while True:
        schedule.run_pending()
        time.sleep(1)
```

## Resultado esperado:

Este laboratorio te ha guiado a través de la implementación de un sistema de clasificación arancelaria automatizada utilizando machine learning, aplicado al sector aduanero. Has aprendido a:

1. Preparar y analizar datos específicos del sector aduanero. <br>
2. Desarrollar y entrenar un modelo de clasificación de códigos HS.<br>
3. Implementar un servicio de predicción.<br>
4. Realizar pruebas y validación del modelo.<br>
5. Documentar el modelo y su uso previsto.<br>
6. Desplegar y monitorear el servicio en producción.<br>
7. Mantener y actualizar el modelo periódicamente.<br>

Para mejorar este proyecto, podrías:

- Implementar técnicas de aprendizaje activo para mejorar continuamente el modelo con la retroalimentación de los agentes aduaneros.<br>
- Integrar el sistema con las bases de datos y sistemas existentes de la aduana.<br>
- Desarrollar una interfaz de usuario amigable para los agentes aduaneros.<br>
- Implementar explicabilidad del modelo (por ejemplo, usando SHAP values) para ayudar a los agentes a entender las predicciones. <br>
- Expandir el sistema para manejar múltiples idiomas y variaciones regionales en las descripciones de productos. <br>

### [Índice](../README.md)

### [Práctica 7. ML Governance (ML + OPS)](../Capítulo7/README.md)
