# Práctica 6. Proceso de despliegue de un modelo de Machine Learning 

## Objetivo de la práctica:

Al finalizar la práctica, serás capaz de:

- Comprender el proceso de despliegue (deployment) de un modelo de Machine Learning.
-	Implementar un servicio de predicción básico.
-	Aprender a monitorear un modelo en producción.
-	Familiarizarse con herramientas y técnicas comunes en el despliegue y monitoreo de modelos ML.

## Duración aproximada:

- 30 minutos.

## Problema:

**Sistema de Recomendación de Películas.**

Imagina que has desarrollado un modelo de recomendación de películas para un servicio de streaming. 

Tu tarea es desplegar este modelo como un servicio de predicción y configurar un sistema de monitoreo para asegurar su correcto funcionamiento en producción.

## Instrucciones 

### Tarea 0. Instalción de dependencias y descarga del dataset.

**Paso 1.** Instala las siguientes librerias para el entorno.

```
pip install pandas numpy scikit-learn flask prometheus-client
```

**Paso 2.** Descarga el conjunto de datos desde el siguiente enlace.

```
https://files.grouplens.org/datasets/movielens/ml-100k.zip
```

**Paso 3.** Una vez descargardo descomprime el archivo.

**Paso 4.** En tu Visual Studio Code crea una carpeta que guardara todos los archivos, ejemplo: `Lab6`.

**Paso 5.** Dentro de la carpeta agrega los siguientes directorios: `datasets/movielens/`.

**Paso 6.** Dentro de la carpeta `movielens` agrega el conjunto de datos descargado y descompreso


### Tarea 1. Preparación del modelo.

Primero, vamos a crear un modelo simple de recomendación. 

**Paso 1.** Crea un archivo `Lab6/train_model.py`:

```
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Cargar los datos
ratings = pd.read_csv('datasets/movielens/ml-100k/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
movies = pd.read_csv('datasets/movielens/ml-100k/u.item', sep='|', encoding='latin-1', names=['item_id', 'title'], usecols=[0, 1])
data = pd.merge(ratings, movies, on='item_id')

# Crear matriz de usuarios y películas
user_movie_matrix = data.pivot_table(index='user_id', columns='title', values='rating').fillna(0)
similarity_matrix = cosine_similarity(user_movie_matrix)

# Guardar modelo
with open('recommendation_model.pkl', 'wb') as f:
    pickle.dump(similarity_matrix, f)
```

**Paso 2.** Ejecuta el script `train_model.py`

```
python train_model.py
```

### Tarea 2. Implementación del Servicio de Predicción.

Ahora, crearemos un servicio web simple utilizando Flask para servir nuestro modelo. 

**Paso 1.** Crea un archivo `Lab6/app.py` y agrega el siguiente codigo:

```
from flask import Flask, request, jsonify
import pickle
from prometheus_client import Counter, start_http_server

# Iniciar métricas
REQUEST_COUNT = Counter('request_count', 'Número de solicitudes al API', ['method', 'endpoint'])

# Configurar Flask
app = Flask(__name__)

# Cargar modelo
with open('recommendation_model.pkl', 'rb') as f:
    similarity_matrix = pickle.load(f)

@app.route('/recommend', methods=['POST'])
def recommend():
    REQUEST_COUNT.labels(method='POST', endpoint='/recommend').inc()
    data = request.json
    user_id = data['user_id']
    recommendations = similarity_matrix[user_id].tolist()
    return jsonify(recommendations)

if __name__ == '__main__':
    start_http_server(8001)  # Servidor para Prometheus
    app.run(debug=True, port=5000)
```

**Paso 2.** Ejecuta el script `app.py`.

**Paso 3.** En una terminal de **Git Bash** ejecuta el siguiente comando para enviar datos de prueba.

```
curl -X POST http://127.0.0.1:5000/recommend -H "Content-Type: application/json" -d '{"user_id": 1}'
```

### Tarea 3. Configuración del monitoreo.

Para el monitoreo, utilizaremos Prometheus para recopilar métricas y Grafana para visualizarlas. 

**Paso 1.** Primero, crea un archivo yaml para definir prometheus en la siguiente ruta: `Lab6/prometheus.yml`

```
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'flask_app'
    static_configs:
      - targets: ['localhost:8001']
```

**Paso 2.** Posteriormente, en la misma ruta crea el siguiente archivo: `Lab6/docker-compose.yml`

```
version: '3.7'
services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
```

**Paso 3.** Ejecuta el comando de docker.

```
docker-compose up -d
```

**NOTA:** Si te sale un error es porque no esta instalado docker, [Download Docker](https://docs.docker.com/desktop/setup/install/windows-install/)

### Tarea 4. Configuración de Grafana.

**Paso 1.** Acceder a Grafana en `http://localhost:3000` en una pestaña nueva de tu navegador:

**Paso 2.** Si te pide usuario y contraseña a todas las opciones puedes escribir `admin`

**Paso 3.** En la pantalla principal de Grafana da clic en la opción **Data Sources**.

**Paso 4.** Selecciona la oción de **Prometheus**.

**Paso 5.** En la sección de **Connection** escribe: `http://prometheus:9090` para enlazar grafana con prometheus.

**Paso 6.** Al final de la pagina da clic en el botón **Save & test**.

**Paso 7.** Ahora hasta arriba esquina superior derecha de la pagina de prometheus en grafana da clic en **Build a Dashboard**.

**Paso 8.** Ahora clic en **Add visualization**

**Paso 9.** Selecciona **Prometheus**

**Paso 10.** En el panel lateral derecho en la opción **Title** y escribe `request_count`


### Tarea 5. Ejecución y pruebas.

**Paso 1.** De vuelta a tu Visual Studio Code en alguna terminal que tengas abierta escribe `python` para activar el shell interactivo.

**Paso 2.** Dentro del shell interactivo escribe el siguiente codigo que lanzara pruebas.

```
import requests

for i in range(10):
    requests.post("http://127.0.0.1:5000/recommend", json={"user_id": i})
```

**Paso 2.** Regresa a tu grafico en Grafana para visualizar tus metricas da clic en la opción **Metric** del panel inferior de tu dashboard y selecciona **scrape_duration_seconds**

**Paso 3.** Un poco mas a la derecha donde seleccionaste la metrica esta la opción **Run queries** da clic y veras la grafica actualizada.

## Resultado esperado:

Este laboratorio te ha introducido a conceptos clave en el despliegue y monitoreo de modelos de ML:

- Implementación de un servicio de predicción con Flask. <br>
- Uso de Prometheus para la recopilación de métricas.<br>
- Visualización de métricas con Grafana.<br>
- Despliegue de servicios utilizando Docker.

Para mejorar este proyecto, podrías:

- Implementar autenticación en el servicio de predicción.<br>
- Agregar más métricas específicas del negocio.<br>
- Configurar alertas basadas en umbrales de métricas.<br>
- Implementar un pipeline de CI/CD para automatizar el despliegue.

### [Índice](../README.md)

### [Práctica 5. Implementación de un modelo de clasificación de imágenes](../Capítulo5/README.md)

### [Práctica 7. ML Governance (ML + OPS)](../Capítulo7/README.md)
