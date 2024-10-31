# Práctica 5. Implementación de un modelo de clasificación de imágenes

## Objetivo de la práctica:

Al finalizar la práctica, serás capaz de:

- Comprender la importancia del testing en proyectos de Machine Learning.
- Aprender a estructurar un proyecto de ML para facilitar el testing y deployment.
- Familiarizarse con el concepto de ML Test Score.
- Implementar un pipeline de Integración Continua (CI) básico.

## Duración aproximada:

- 30 minutos.

## Estructura del proyecto:


## Instrucciones 

Antes de comenzar, es importante entender cómo estructurar un proyecto de Machine Learning. Una estructura típica podría ser:

```
proyecto_ml/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data/
│   │   └── make_dataset.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── train_model.py
│   │   └── predict_model.py
│   └── visualization/
│       └── visualize.py
│
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   └── test_model.py
│
├── notebooks/
│   └── exploratory_data_analysis.ipynb
│
├── requirements.txt
├── setup.py
└── README.md
```

## Problema a desarrollar: 

**Implementación de un modelo de clasificación de imágenes.**

Imagina que estás trabajando en un proyecto para clasificar imágenes de perros y gatos. Tu tarea es implementar el modelo, crear tests para asegurar su calidad y configurar un pipeline de CI básico.

## Tarea 1. Implementación del modelo.

Primero, implementaremos un modelo simple usando TensorFlow/Keras. 

**Paso 1.** Crea un archivo `src/models/train_model.py`:

```python
import tensorflow as tf
from tensorflow.keras import layers, models
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
def train_model(model, train_data, validation_data):
    history = model.fit(train_data, epochs=10, validation_data=validation_data)
    return history
if __name__ == "__main__":
     Aquí iría el código para cargar los datos y entrenar el modelo
    pass
```
### Tarea 2. Implementación de tests.

Ahora, crearemos algunos tests básicos. 

**Paso 1.** Crea un archivo `tests/test_model.py`:

```python
import unittest
import tensorflow as tf
from src.models.train_model import create_model
class TestModel(unittest.TestCase):
    def test_model_structure(self):
        model = create_model()
        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(len(model.layers), 8)
    
    def test_model_output_shape(self):
        model = create_model()
        test_input = tf.random.normal((1, 150, 150, 3))
        output = model(test_input)
        self.assertEqual(output.shape, (1, 1))
if __name__ == '__main__':
    unittest.main()
```
### Tarea 3. ML Test Score.

El ML Test Score es una métrica para evaluar la madurez y la calidad de un sistema de ML. Durante esta tarea, implementaremos una versión simplificada. 

**Paso 1.** Crea un archivo `src/evaluation/ml_test_score.py`:

```python
def calculate_ml_test_score(model, data_tests, model_tests, monitoring):
    score = 0
    if all(data_tests):
        score += 1
    if all(model_tests):
        score += 1
    if monitoring:
        score += 1
    return score / 3   Normalizado a un rango de 0 a 1
```

#### Ejemplo de uso

```
data_tests = [True, True, False]   Ej: tests de integridad de datos, distribución, etc.
model_tests = [True, True]   Ej: tests de rendimiento, sesgo, etc.
monitoring = True   Ej: monitoreo en producción
ml_score = calculate_ml_test_score(None, data_tests, model_tests, monitoring)
print(f"ML Test Score: {ml_score}")
```

### Tarea 4. Configuración de CI.

Finalmente, configuraremos un pipeline de CI básico usando GitHub Actions. 

**Paso 1.** Crea un archivo `.github/workflows/ci.yml`:

```yaml
name: CI
on: [push]
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
      run: python -m unittest discover tests
    - name: Calculate ML Test Score
      run: python src/evaluation/ml_test_score.py
```

## Solución paso a paso:
 
1. **Implementación del Modelo**:
   - Creamos una arquitectura de red neuronal convolucional simple para clasificación binaria.
   - Definimos funciones para crear y entrenar el modelo.
2. **Implementación de Tests**:
   - Creamos tests unitarios para verificar la estructura del modelo y la forma de salida.
   - Estos tests aseguran que el modelo se crea correctamente y produce salidas del tamaño esperado.
3. **ML Test Score**:
   - Implementamos una función simple para calcular el ML Test Score.
   - Este score evalúa aspectos de calidad de datos, tests del modelo y monitoreo.
4. **Configuración de CI**:
   - Configuramos un workflow de GitHub Actions que se ejecuta en cada push.
   - El workflow instala dependencias, ejecuta tests y calcula el ML Test Score.

## Resultado esperado:

Este laboratorio te ha introducido a conceptos clave en testing y deployment de modelos de ML:

- Estructuración de proyectos de ML.
- Implementación de tests unitarios para modelos.
- Cálculo de ML Test Score.
- Configuración básica de CI.

Para mejorar este proyecto, podrías:

- Agregar más tests (ej: tests de datos, tests de integración).
- Implementar validación cruzada en el entrenamiento del modelo.
- Añadir un paso de deployment automático en el pipeline de CI.
- Implementar logging y monitoreo más avanzados.

### [Índice](../README.md)

### [Práctica 4. Clasificación de imágenes de dígitos escritos a mano](../Capítulo4/README.md)

### [Práctica 6. Proceso de despliegue de un modelo de Machine Learning](../Capítulo6/README.md)
