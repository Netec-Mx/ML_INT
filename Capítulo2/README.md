# Práctica 2. Mejora de recomendaciones de productos 

## Objetivo de la práctica:

Al finalizar la práctica, serás capaz de:

- Trabajar con fuentes de datos y realizar etiquetado. <br>
- Implementar almacenamiento y versionamiento de datos. <br>
- Procesar datos utilizando PySpark.<br>
- Crear y testear pipelines de datos.

## Duración aproximada:

- 60 minutos.

## Problema a desarrollar:

Una empresa de comercio electrónico quiere mejorar sus recomendaciones de productos utilizando machine learning. Para ello, necesitan procesar y analizar sus datos de interacciones de usuarios con productos. 

Tu tarea es crear un pipeline de ingeniería de datos que prepare estos datos para su uso en un modelo de recomendación.

## Instrucciones:

### Tarea 1.1. Fuentes y etiquetado de datos.

**Paso 1.** Crear un dataset de ejemplo.

Primero, vamos a crear un dataset de ejemplo que simule las interacciones de usuarios con productos.

**Paso 2.** Crea un archivo llamado `generate_data.py` con el siguiente contenido:

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
def generate_sample_data(n_users=1000, n_products=100, n_interactions=10000):
    np.random.seed(42)
    
     Generar usuarios
    users = pd.DataFrame({
        'user_id': range(1, n_users + 1),
        'age': np.random.randint(18, 70, n_users),
        'gender': np.random.choice(['M', 'F'], n_users)
    })
    
     Generar productos
    products = pd.DataFrame({
        'product_id': range(1, n_products + 1),
        'category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], n_products),
        'price': np.random.uniform(10, 1000, n_products).round(2)
    })
    
     Generar interacciones
    interactions = pd.DataFrame({
        'user_id': np.random.choice(users['user_id'], n_interactions),
        'product_id': np.random.choice(products['product_id'], n_interactions),
        'timestamp': [datetime.now() - timedelta(days=np.random.randint(0, 365)) for _ in range(n_interactions)],
        'interaction_type': np.random.choice(['view', 'cart', 'purchase'], n_interactions, p=[0.7, 0.2, 0.1])
    })
    
    return users, products, interactions
 Generar datos
users, products, interactions = generate_sample_data()
 Guardar datos en archivos CSV
users.to_csv('users.csv', index=False)
products.to_csv('products.csv', index=False)
interactions.to_csv('interactions.csv', index=False)
print("Datos de ejemplo generados y guardados en archivos CSV.")
```

**Paso 2.** Ejecuta el siguiente script para generar los datos de ejemplo:

```
python generate_data.py
```

### Tarea 1.2. Etiquetar los datos.

Para nuestro problema de recomendación, vamos a etiquetar las interacciones como _"positivas"_ si son compras y _"negativas"_ en caso contrario. Crea un nuevo script llamado `label_data.py`:

```python
import pandas as pd
def label_interactions(interactions_df):
    interactions_df['label'] = (interactions_df['interaction_type'] == 'purchase').astype(int)
    return interactions_df
 Cargar datos de interacciones
interactions = pd.read_csv('interactions.csv')
 Etiquetar datos
labeled_interactions = label_interactions(interactions)
 Guardar datos etiquetados
labeled_interactions.to_csv('labeled_interactions.csv', index=False)
print("Datos etiquetados y guardados en labeled_interactions.csv")
```

Ejecuta el siguiente script para etiquetar los datos:

```
python label_data.py
```

### Tarea 2. Almacenamiento y versionamiento.

**Paso 1.** Para el almacenamiento y versionamiento de datos, utilizarás DVC (Data Version Control). Para inciar, instala DVC:

```
pip install dvc
```

**Paso 2.** Inicializa un repositorio Git y DVC.

```
git init
dvc init
```

**Paso 3.** Ahora, agrega los archivos CSV al control de versiones de DVC:

```
dvc add users.csv products.csv labeled_interactions.csv
git add .gitignore users.csv.dvc products.csv.dvc labeled_interactions.csv.dvc
git commit -m "Add initial datasets"
```

### Tarea 3. Procesamiento de datos con PySpark.

**Paso 1.** Ahora vamos a procesar los datos utilizando PySpark. Primero, instala PySpark:

```
pip install pyspark
```

**Paso 2.** Crea un nuevo archivo llamado `process_data.py`:

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
def process_data(spark):
     Cargar datos
    users = spark.read.csv('users.csv', header=True, inferSchema=True)
    products = spark.read.csv('products.csv', header=True, inferSchema=True)
    interactions = spark.read.csv('labeled_interactions.csv', header=True, inferSchema=True)
     Unir datos
    data = interactions.join(users, on='user_id').join(products, on='product_id')
     Procesar datos
    processed_data = data.withColumn(
        'age_group',
        when(col('age') < 30, 'young')
        .when((col('age') >= 30) & (col('age') < 50), 'middle')
        .otherwise('senior')
    ).withColumn(
        'price_category',
        when(col('price') < 50, 'low')
        .when((col('price') >= 50) & (col('price') < 200), 'medium')
        .otherwise('high')
    )
     Seleccionar columnas relevantes
    final_data = processed_data.select(
        'user_id', 'product_id', 'age_group', 'gender', 'category', 
        'price_category', 'interaction_type', 'label'
    )
    return final_data
if __name__ == "__main__":
    spark = SparkSession.builder.appName("DataProcessing").getOrCreate()
    
    processed_data = process_data(spark)
    
     Guardar datos procesados
    processed_data.write.csv('processed_data.csv', header=True, mode='overwrite')
    
    spark.stop()
print("Datos procesados y guardados en processed_data.csv")
```

**Paso 3.** Ejecuta el siguiente script para procesar los datos:

```
python process_data.py
```

 ### Tarea 4. Testing de pipelines de datos.
 
**Paso 1.** Para asegurar la calidad de nuestro pipeline de datos, vamos a crear algunas pruebas unitarias. Crea un archivo llamado `test_data_pipeline.py`:

```python
import unittest
from pyspark.sql import SparkSession
from process_data import process_data
class TestDataPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.appName("TestDataProcessing").getOrCreate()
    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()
    def test_process_data(self):
        processed_data = process_data(self.spark)
         Verificar que el DataFrame no esté vacío
        self.assertTrue(processed_data.count() > 0)
         Verificar que todas las columnas esperadas estén presentes
        expected_columns = {'user_id', 'product_id', 'age_group', 'gender', 'category', 
                            'price_category', 'interaction_type', 'label'}
        self.assertEqual(set(processed_data.columns), expected_columns)
         Verificar que los valores de 'age_group' sean correctos
        age_groups = processed_data.select('age_group').distinct().collect()
        self.assertEqual(set([row['age_group'] for row in age_groups]), {'young', 'middle', 'senior'})
         Verificar que los valores de 'price_category' sean correctos
        price_categories = processed_data.select('price_category').distinct().collect()
        self.assertEqual(set([row['price_category'] for row in price_categories]), {'low', 'medium', 'high'})
         Verificar que los valores de 'label' sean 0 o 1
        labels = processed_data.select('label').distinct().collect()
        self.assertEqual(set([row['label'] for row in labels]), {0, 1})
if __name__ == '__main__':
    unittest.main()
```

**Paso 2.** Ejecuta las pruebas:

```
python -m unittest test_data_pipeline.py
```

## Resultado esperado:

En este laboratorio, has aprendido a:
1. Crear y etiquetar datos de ejemplo para un problema de recomendación.
2. Utilizar DVC para el almacenamiento y versionamiento de datos.
3. Procesar datos utilizando PySpark, realizando transformaciones y uniones de datos.
4. Crear pruebas unitarias para verificar la calidad del pipeline de datos.

Este pipeline de ingeniería de datos ha preparado los datos para su uso en un modelo de recomendación de productos. Los próximos pasos serían utilizar estos datos procesados para entrenar y evaluar un modelo de machine learning.

### [Índice](../README.md)

### [Práctica 1. Análisis de ventas](../Capítulo1/README.md)

### [Práctica 3. Mejora del proceso de aprobación de créditos](../Capítulo3/README.md)
