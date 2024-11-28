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

**Paso 3.** Crea un directorio llamado **Lab8**

**Paso 4.** Ahora crea un archivo CSV (`data/import_data.csv`) con la siguiente estructura:

```
id,description,hs_code
1,"Laptop computer, 15-inch screen, 8GB RAM, 256GB SSD",8471.30
2,"Men's cotton t-shirt, short sleeve, blue",6109.10
3,"Smartphone, 6.1-inch display, 128GB storage",8517.13
```

### Tarea 2. Preparación y análisis de datos.

**Paso 1.** Crea un script (`src/data_preparation.py`) para preparar los datos, sino existe la carpeta **src** debes crearla primero:

```
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def prepare_data(file_path):
    # Cargar datos
    data = pd.read_csv(file_path)
    
    # Convertir los códigos arancelarios en categorías (str)
    data['hs_code'] = data['hs_code'].astype(str)
    
    # Dividir en características (X) y etiquetas (y)
    X = data['description']
    y = data['hs_code']
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Vectorizar las descripciones usando TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
    # Crear directorios si no existen
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Guardar el vectorizador y los datos procesados
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.joblib')
    joblib.dump(X_train_vectorized, 'data/processed/X_train.joblib')
    joblib.dump(X_test_vectorized, 'data/processed/X_test.joblib')
    joblib.dump(y_train, 'data/processed/y_train.joblib')
    joblib.dump(y_test, 'data/processed/y_test.joblib')

if __name__ == "__main__":
    prepare_data('data/import_data.csv')
```

**Paso 2.** Ejecuta el escript

```
python src/data_preparation.py
```

### Tarea 3. Desarrollo del modelo.

**Paso 1.** Crea un script (`src/train_model.py`) para entrenar el modelo:

```
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import mlflow

def train_and_evaluate_model():
    # Cargar datos procesados
    X_train = joblib.load('data/processed/X_train.joblib')
    X_test = joblib.load('data/processed/X_test.joblib')
    y_train = joblib.load('data/processed/y_train.joblib')
    y_test = joblib.load('data/processed/y_test.joblib')
    
    # Configurar MLflow
    mlflow.set_experiment("HS Code Classification")
    
    with mlflow.start_run():
        # Entrenar el modelo
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        model.fit(X_train, y_train)
        
        # Evaluar el modelo
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Registrar métricas específicas
        mlflow.log_metric("accuracy", report['accuracy'] if 'accuracy' in report else 0.0)
        mlflow.log_metric("weighted_avg_f1-score", report['weighted avg']['f1-score'])
        
        # Crear el directorio models si no existe
        os.makedirs('models', exist_ok=True)
        
        # Guardar el modelo
        joblib.dump(model, 'models/hs_code_classifier.joblib')
        print("Entrenamiento completado. Modelo guardado.")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

if __name__ == "__main__":
    train_and_evaluate_model()
```

**Paso 2.** Ejecuta el escript

```
python src/train_model.py
```

### Tarea 4. Implementación del servicio de predicción.

**Paso 1.** Crea un script (`src/predict_service.py`) para implementar el servicio de predicción:

```
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Cargar el modelo y el vectorizador
model = joblib.load('models/hs_code_classifier.joblib')
vectorizer = joblib.load('models/tfidf_vectorizer.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    description = data.get('description', '')
    
    # Vectorizar la descripción
    vectorized_description = vectorizer.transform([description])
    
    # Realizar la predicción
    prediction = model.predict(vectorized_description)[0]
    return jsonify({'hs_code': prediction})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

**Paso 2.** Ejecuta el escript

```
python src/predict_service.py
```

**Paso 3.** En una terminal **Git Bash** ejecuta el siguiente comando para enviar solicitudes.

```
curl -X POST http://localhost:5000/predict \
-H "Content-Type: application/json" \
-d '{"description": "Smartphone, 6.5-inch display, 128GB storage"}'
```

```
curl -X POST http://localhost:5000/predict \
-H "Content-Type: application/json" \
-d "{\"description\": \"Men's cotton t-shirt, short sleeve, blue\"}"
```

### Tarea 5. Pruebas y validación.

**Paso 1.** Crea un script (`src/test_model.py`) para realizar pruebas adicionales:

```
import os
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def test_model():
    # Cargar el modelo y los datos de prueba
    model = joblib.load('models/hs_code_classifier.joblib')
    X_test = joblib.load('data/processed/X_test.joblib')
    y_test = joblib.load('data/processed/y_test.joblib')
    
    # Realizar predicciones
    y_pred = model.predict(X_test)
    
    # Generar informe de clasificación
    report = classification_report(y_test, y_pred, zero_division=0)
    print("Classification Report:")
    print(report)
    
    # Generar matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Crear el directorio 'reports/' si no existe
    os.makedirs('reports', exist_ok=True)
    
    # Guardar la imagen de la matriz de confusión
    plt.savefig('reports/confusion_matrix.png')
    plt.close()

if __name__ == "__main__":
    test_model()
```

**Paso 2.** Ejecuta el escript

```
python src/test_model.py
```

### Tarea 6. Documentación.

**Paso 1.** Crea un documento de modelo (`src/model_card.md`):

```
# Model Card: HS Code Classifier

## Detalles del modelo
- **Developer:** [Tu nombre]
- **Model Date:** [Fecha actual]
- **Model Version:** 1.0
- **Model Type:** Random Forest Classifier

## Uso
- **Primary Use:** Ayudar a los agentes aduaneros a clasificar productos importados según el Sistema Armonizado (HS).
- **Intended Users:** Agentes aduaneros y especialistas en comercio exterior.

## Datos de entrenamiento
- **Source:** Datos históricos de importación con descripciones de productos y sus códigos HS correspondientes.
- **Size:** [Número de registros].
- **Preprocessing:** Vectorización TF-IDF de descripciones de productos.

## Datos de evaluación
- **Size:** 20% del conjunto de datos original como prueba.

## Consideraciones eticas
- El modelo debe ser usado como herramienta de apoyo, no como reemplazo de la toma de decisiones humanas.
- Auditorías regulares para evitar sesgos en los datos.

## Recomendaciones
- El rendimiento puede variar para categorías de productos con representación limitada en los datos de entrenamiento.
- Se recomienda actualizar el modelo periódicamente con datos nuevos.
```
 
### Tarea 7. Monitoreo del servicio.

**Paso 1.** Activa el software de **Docker Desktop** en el sistema operativo de Windows o tu area de trabajo.

**Paso 2.** En la raíz de tu proyecto, crea un archivo llamado `prometheus.yml` con el siguiente contenido:

```
global:
  scrape_interval: 15s  # Frecuencia con la que se recolectan métricas

scrape_configs:
  - job_name: 'hs_code_service'
    static_configs:
      - targets: ['host.docker.internal:8000']  # Endpoint del servicio de métricas
```

**Paso 3.** En la raíz del proyecto crea un archivo `docker-compose.yml` para gestionar **Prometheus** y **Grafana** con Docker:

```
version: '3.8'
services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"  # Exponer Prometheus en el puerto 9090
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
  
  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"  # Exponer Grafana en el puerto 3000
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

**Paso 4.** Ejecuta el siguiente comando en el directorio donde está el archivo **docker-compose.yml**:

```
docker-compose up -d
```

**TROUBLESHOOTING:** Si ya tienes creados los contendores puedes eliminarlos y recrearlos aplicando los siguientes comandos y si la salida de los comandos aparece vacia ejecuta el **Paso 4**.

```
docker ps -a
```
```
docker rm <CONTAINER-ID> -f
```
```
docker ps
```
```
docker ps -a
```

**Paso 5.** Crea un script (`src/monitor_service.py`) para monitorear el servicio:

```
import requests
import time
import logging
from prometheus_client import start_http_server, Counter, Histogram

# Configurar logging
logging.basicConfig(filename='hs_code_classifier.log', level=logging.INFO)

# Métricas de Prometheus
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
        
        time.sleep(60)  # Esperar 1 minuto antes de la próxima verificación
if __name__ == '__main__':
    start_http_server(8000)  # Iniciar servidor de métricas de Prometheus
    monitor_prediction_service()
```

**Paso 6.** Ejecuta el escript y deja la ventana activa, no la cierres

```
python src/monitor_service.py
```

**Paso 7.** Sino tienes activo el script **predict_service.py** activalo en una terminal de Visual Studio Code y realiza las pruebas, tambien deja la terminal abierta.

**Paso 8.** Acceder a Grafana en `http://localhost:3000` en una pestaña nueva de tu navegador:

**Paso 9.** Si te pide usuario y contraseña a todas las opciones puedes escribir `admin`

**Paso 10.** En la pantalla principal de Grafana da clic en la opción **Data Sources**.

**Paso 11.** Selecciona la oción de **Prometheus**.

**Paso 12.** En la sección de **Connection** escribe: `http://prometheus:9090` para enlazar grafana con prometheus.

**Paso 13.** Al final de la pagina da clic en el botón **Save & test**.

**Paso 14.** Ahora hasta arriba esquina superior derecha de la pagina de prometheus en grafana da clic en **Build a Dashboard**.

**Paso 15.** Ahora clic en **Add visualization**

**Paso 16.** Selecciona **Prometheus**

**Paso 17.** En el panel lateral derecho en la opción **Title** y escribe `HS Code Monitoring`

**Paso 18.** Selecciona la metrica llamada `hs_code_predictions_total` o `prediction_response_time_seconds_sum`, tambien puedes probar cualquier otra de interes.

### Tarea 8. Actualización del modelo.

**Paso 1.** Crea un script (`src/update_model.py`) para actualizar periódicamente el modelo:

```
import schedule
import time
from datetime import datetime
from train_model import train_and_evaluate_model

def update_model_job():
    today = datetime.now()
    # Verificar si es el primer día del mes
    if today.day == 1:
        print("Updating HS Code Classifier model...")
        train_and_evaluate_model()
        print("Model updated and saved.")

# Programar la verificación diaria
schedule.every().day.at("02:00").do(update_model_job)

if __name__ == "__main__":
    while True:
        schedule.run_pending()
        time.sleep(1)
```

**Paso 2.** Ejecuta el escript, el escript quedara en proceso de espera para la programación de la actualización del modelo, para esta tarea solo es demostrativo.

```
python src/update_model.py
```

```
CTRL + C
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
