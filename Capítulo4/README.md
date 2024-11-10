# Práctica 4. Clasificación de imágenes de dígitos escritos a mano 

## Objetivo de la práctica:

Al finalizar la práctica, serás capaz de:

- Implementar un modelo de red neuronal para clasificación de imágenes.
- Aplicar técnicas de debugging para identificar y resolver problemas en el entrenamiento.
- Evaluar el rendimiento del modelo y proponer mejoras.
- Realizar _hipertuning_ para optimizar el rendimiento del modelo.

## Duración aproximada:

- 60 minutos.

## Introducción:

En este laboratorio, abordaremos el proceso de entrenamiento, debugging, evaluación y mejora de un modelo de machine learning. Nos centraremos en un problema de clasificación de imágenes de dígitos escritos a mano, utilizando el conjunto de datos MNIST.

## Problema a desarrollar:

Desarrollar un modelo de clasificación para reconocer dígitos escritos a mano (0-9) a partir de imágenes en escala de grises de 28 x 28 píxeles.

## Instrucciones:

### Tarea 1. Preparación del entorno y datos.

**Paso 1.** Primero, vamos a instalar las bibliotecas necesarias y preparar nuestros datos.

**NOTA:** Abre la aplicación de **Microsoft Store** y busca el software de **python** elige la version **Python 3.10** e instalala en la maquina virtual. Reinicia Visual Studio Code.

```
py -3.10 -m venv tf-env
.\tf-env\Scripts\activate
pip install tensorflow numpy matplotlib scikit-learn seaborn
```

**Paso 2.** Ahora, crea un archivo `prepare_data.py`:

```
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def load_and_prepare_data():
    try:
        # Cargar el conjunto de datos MNIST
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Normalizar los datos
        X_train = X_train.astype('float32') / 255
        X_test = X_test.astype('float32') / 255

        # Reshape para que sea compatible con la entrada de la red neuronal
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

        # Guardar los datos procesados
        np.save('X_train.npy', X_train)
        np.save('X_test.npy', X_test)
        np.save('y_train.npy', y_train)
        np.save('y_test.npy', y_test)
        
        print("Datos preparados y guardados correctamente.")
        return X_train, y_train
    except Exception as e:
        print(f"Ocurrió un error al preparar los datos: {e}")
        return None, None

def visualize_samples(X, y, num_samples=10, filename="ejemplos_mnist.png"):
    try:
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        for i, ax in enumerate(axes.flat):
            ax.imshow(X[i].reshape(28, 28), cmap='gray')
            ax.set_title(f"Dígito: {y[i]}")
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Ejemplos visualizados en '{filename}'.")
    except Exception as e:
        print(f"Ocurrió un error al visualizar las imágenes: {e}")

# Ejecutar funciones
X_train, y_train = load_and_prepare_data()
if X_train is not None and y_train is not None:
    visualize_samples(X_train, y_train)
```
**Paso 2.** Ejecuta el script para preparar los datos:

```
python prepare_data.py
```

### Tarea 2. Implementación inicial del modelo.

**Paso 1.** Crea un archivo `train_model.py`:

```
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Cargar datos
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Definir el modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer=Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2, batch_size=128)

# Evaluar el modelo
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Precisión en el conjunto de prueba: {test_acc:.4f}")

# Guardar el modelo
model.save('mnist_model.h5')
print("Modelo entrenado y guardado como 'mnist_model.h5'.")
```

**Paso 2.** Ejecuta el script siguiente para entrenar el modelo inicial:

```
python train_model.py
```

### Tarea 2. Debugging.

Ahora, vamos a implementar algunas técnicas de debugging para identificar posibles problemas en el entrenamiento. 

**Paso 1.** Crea un archivo `debug_model.py`:

```
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Cargar datos y modelo
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')
model = load_model('mnist_model.h5')

#Visualizar la distribución de las predicciones
y_pred = model.predict(X_test)
plt.hist(np.argmax(y_pred, axis=1), bins=10)
plt.title('Distribución de predicciones')
plt.xlabel('Dígito')
plt.ylabel('Frecuencia')
plt.savefig('distribucion_predicciones.png')
plt.close()

# Visualizar la matriz de confusión
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(y_test, np.argmax(y_pred, axis=1))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.savefig('matriz_confusion.png')
plt.close()

# Analizar ejemplos mal clasificados
misclassified = np.where(np.argmax(y_pred, axis=1) != y_test)[0]
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    if i < len(misclassified):
        idx = misclassified[i]
        ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
        ax.set_title(f"Real: {y_test[idx]}, Pred: {np.argmax(y_pred[idx])}")
        ax.axis('off')
plt.tight_layout()
plt.savefig('ejemplos_mal_clasificados.png')
plt.close()
print("Análisis de debugging completado. Revisa las imágenes generadas.")
```

**Paso 2.** Ejecuta el script de debugging:

```
python debug_model.py
```

### Tarea 4. Evaluación y mejora.

Basándonos en los resultados del debugging, vamos a implementar algunas mejoras. 

**Paso 1.** Crea un archivo `improve_model.py`:

```
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Cargar datos
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Definir el modelo mejorado
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    BatchNormalization(),
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Definir callbacks
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
lr_reducer = ReduceLROnPlateau(factor=0.5, patience=5)

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, batch_size=128,
                    callbacks=[early_stopping, lr_reducer])

# Evaluar el modelo
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Precisión en el conjunto de prueba: {test_acc:.4f}")

# Guardar el modelo mejorado
model.save('mnist_model_improved.h5')
print("Modelo mejorado entrenado y guardado como 'mnist_model_improved.h5'.")
```

**Paso 2.** Ejecuta el script para entrenar el modelo mejorado:

```
python improve_model.py
```

### Tarea 5. Hipertuning.
 
Finalmente, vamos a realizar un ajuste de hiperparámetros utilizando Keras Tuner. 

**Paso 1.** Crea un archivo `hypertune_model.py`:

```
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

# Cargar datos
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

def build_model(hp):
    model = Sequential()
    model.add(Conv2D(hp.Int('conv1_units', min_value=32, max_value=128, step=32),
                     (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    
    for i in range(hp.Int('n_conv_layers', 1, 3)):
        model.add(Conv2D(hp.Int(f'conv{i+2}_units', min_value=32, max_value=128, step=32),
                         (3, 3), activation='relu'))
        model.add(BatchNormalization())
    
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    
    for i in range(hp.Int('n_dense_layers', 1, 3)):
        model.add(Dense(hp.Int(f'dense{i+1}_units', min_value=64, max_value=512, step=64),
                        activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float(f'dropout{i+1}', min_value=0.0, max_value=0.5, step=0.1)))
    
    model.add(Dense(10, activation='softmax'))
    
    model.compile(optimizer=Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    directory='mnist_tuning',
    project_name='mnist_classification'
)
tuner.search(X_train, y_train, epochs=20, validation_split=0.2)
best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Mejores hiperparámetros encontrados:")
print(best_hyperparameters.values)

# Evaluar el mejor modelo
test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
print(f"Precisión en el conjunto de prueba con el mejor modelo: {test_acc:.4f}")

# Guardar el mejor modelo
best_model.save('mnist_model_best.h5')
print("Mejor modelo guardado como 'mnist_model_best.h5'.")
```

**Paso 2.** Ejecuta el script para realizar el hipertuning:

```
python hypertune_model.py
```

## Resultado esperado:

En este laboratorio, has aprendido a:

1. Preparar y visualizar datos para un problema de clasificación de imágenes. <br>
2. Implementar y entrenar un modelo de red neuronal convolucional.<br>
3. Aplicar técnicas de debugging para identificar problemas en el entrenamiento.<br>
4. Evaluar el rendimiento del modelo y proponer mejoras basadas en el análisis.<br>
5. Realizar hipertuning para optimizar los hiperparámetros del modelo.

Estos pasos son fundamentales en el proceso de desarrollo y refinamiento de modelos de machine learning, permitiéndote crear modelos más precisos y robustos.

Para continuar mejorando, podrías considerar:

- Implementar técnicas de aumento de datos para mejorar la generalización del modelo.<br>
- Explorar arquitecturas de redes neuronales más avanzadas, como las redes residuales.<br>
- Investigar técnicas de interpretabilidad del modelo para entender mejor sus predicciones.<br>
- Implementar un sistema de monitoreo continuo del rendimiento del modelo en producción.

### [Índice](../README.md)

### [Práctica 3. Mejora del proceso de aprobación de créditos](../Capítulo3/README.md)

### [Práctica 5. Implementación de un modelo de clasificación de imágenes](../Capítulo5/README.md)
