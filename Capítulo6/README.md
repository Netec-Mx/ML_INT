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

### Tarea 1. Preparación del modelo.

Primero, vamos a crear un modelo simple de recomendación. 

**Paso 1.** Crea un archivo`src/models/movie_recommender.py`:

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
class MovieRecommender:
    def __init__(self):
        self.movies_df = None
        self.tfidf_matrix = None
        
    def fit(self, movies_data):
        self.movies_df = pd.DataFrame(movies_data)
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.movies_df['description'])
        
    def get_recommendations(self, movie_id, top_n=5):
        idx = self.movies_df.index[self.movies_df['id'] == movie_id].tolist()[0]
        sim_scores = list(enumerate(cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix)[0]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]
        movie_indices = [i[0] for i in sim_scores]
        return self.movies_df['title'].iloc[movie_indices].tolist()
    def save_model(self, filename):
        joblib.dump(self, filename)
    @classmethod
    def load_model(cls, filename):
        return joblib.load(filename)
```

**Ejemplo de uso:**

```
if __name__ == "__main__":
    movies_data = [
        {'id': 1, 'title': 'The Shawshank Redemption', 'description': 'Two imprisoned men bond over a number of years...'},
        {'id': 2, 'title': 'The Godfather', 'description': 'The aging patriarch of an organized crime dynasty...'},
        {'id': 3, 'title': 'The Dark Knight', 'description': 'When the menace known as the Joker emerges from his mysterious past...'},
         ... más películas ...
    ]
    
    recommender = MovieRecommender()
    recommender.fit(movies_data)
    recommender.save_model('movie_recommender.joblib')
```

### Tarea 2. Implementación del Servicio de Predicción.

Ahora, crearemos un servicio web simple utilizando Flask para servir nuestro modelo. 

**Paso 1.** Crea un archivo `src/app.py`:

```python
from flask import Flask, request, jsonify
from models.movie_recommender import MovieRecommender
app = Flask(__name__)
 Cargar el modelo al iniciar la aplicación
model = MovieRecommender.load_model('movie_recommender.joblib')
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    movie_id = data['movie_id']
    recommendations = model.get_recommendations(movie_id)
    return jsonify({'recommendations': recommendations})
if __name__ == '__main__':
    app.run(debug=True)
```

### Tarea 3. Configuración del monitoreo.

Para el monitoreo, utilizaremos Prometheus para recopilar métricas y Grafana para visualizarlas. 

**Paso 1.** Primero, instala las dependencias necesarias:

```
pip install prometheus-client flask-prometheus-metrics
```

**Paso 2.** Posteriormente, actualiza nuestro `src/app.py` para incluir métricas:

```python
from flask import Flask, request, jsonify
from models.movie_recommender import MovieRecommender
from prometheus_client import Counter, Histogram
from flask_prometheus_metrics import register_metrics
app = Flask(__name__)
 Métricas
RECOMMENDATIONS = Counter('recommendations_total', 'Total number of recommendations made')
RESPONSE_TIME = Histogram('recommendation_response_time_seconds', 'Response time for recommendations')
 Registrar métricas
register_metrics(app)

 Cargar el modelo al iniciar la aplicación
model = MovieRecommender.load_model('movie_recommender.joblib')
@app.route('/recommend', methods=['POST'])
@RESPONSE_TIME.time()
def recommend():
    data = request.json
    movie_id = data['movie_id']
    recommendations = model.get_recommendations(movie_id)
    RECOMMENDATIONS.inc()
    return jsonify({'recommendations': recommendations})
if __name__ == '__main__':
    app.run(debug=True)
```

### Tarea 4. Configuración de Prometheus.

**Paso 1.** Crea un archivo `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
scrape_configs:
  - job_name: 'flask'
    static_configs:
      - targets: ['localhost:5000']
```
### Tarea 5. Configuración de Grafana.

**Paso 1.** Instala Grafana siguiendo las instrucciones oficiales para tu sistema operativo. <br>
**Paso 2.** Configura Prometheus como fuente de datos en Grafana. <br>
**Paso 3.** Crea un dashboard con gráficos para las métricas `recommendations_total` y `recommendation_response_time_seconds`.

### Tarea 6. Despliegue.

Para desplegar nuestro servicio, utilizaremos Docker. 

**Paso 1.** Crea un `Dockerfile`:

```dockerfile
FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ .
COPY movie_recommender.joblib .
CMD ["python", "app.py"]
```

**Paso 2.** Crea un archivo `docker-compose.yml`:

```yaml
version: '3'
services:
  recommender:
    build: .
    ports:
      - "5000:5000"
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
```

 ### Tarea 7. Ejecución y pruebas.

**Paso 1.** Construye y ejecuta los contenedores:

   ```
   docker-compose up --build
   ```

**Paso 2.** Prueba el servicio de recomendación:

   ```
   curl -X POST -H "Content-Type: application/json" -d '{"movie_id": 1}' http://localhost:5000/recommend
   ```

**Paso 3.** Accede a Grafana en `http://localhost:3000` y configura el dashboard para visualizar las métricas.

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
