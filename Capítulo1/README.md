# Práctica 1. Análisis de ventas

## Objetivo de la práctica:

Al finalizar la práctica, serás capaz de:

- Implementar Programación Orientada a Objetos (POO) en Python. <br>
- Utilizar módulos y clases en Python.<br>
- Asegurar la mantenibilidad y documentación del código.<br>
- Implementar *testing unitario*.

## Duración aproximada:

- 60 minutos.

## Datos de entrada:

Los datos de ventas se proporcionarán en un archivo de texto (ventas_mensuales.txt) con el siguiente formato:

```
fecha,monto
2024-01-01,1500.50
2024-01-02,2300.75
2024-01-03,1800.25
...
```

## Requerimientos funcionales:

1. Cargar datos de ventas desde un archivo de texto. <br>
2. Calcular el total de ventas del mes.<br>
3. Calcular el promedio de ventas diarias.<br>
4. Identificar el día con mayores ventas.<br>
5. Generar un informe con estas estadísticas.<br>

## Instrucciones: 

### **Tarea 1.** Crear la estructura del proyecto.

Paso 1. Crea un nuevo directorio llamado `analizador_ventas`.

Paso 2. Dentro de este directorio, crea los siguientes archivos:
   - `analizador.py`: Contendrá la clase principal `AnalizadorVentas`.
   - `test_analizador.py`: Contendrá los tests unitarios.
   - `main.py`: Será el punto de entrada de nuestra aplicación.
   - `ventas_mensuales.txt`: Un archivo de texto con datos de ventas de ejemplo.

### **Tarea 2.** Implementar la clase _AnalizadorVentas_.

**Paso 1.** En el archivo `analizador.py`, implementa la siguiente clase:

```python
from datetime import datetime
class AnalizadorVentas:
    """
    Clase para analizar datos de ventas mensuales.
    """
    def __init__(self):
        """
        Inicializa la clase AnalizadorVentas.
        """
        self.ventas = []
    def cargar_datos(self, nombre_archivo):
        """
        Carga datos de ventas desde un archivo de texto.
        Args:
            nombre_archivo (str): Nombre del archivo a cargar.
        """
        self.ventas = []
        with open(nombre_archivo, 'r') as archivo:
            next(archivo)   Saltar la primera línea (encabezados)
            for linea in archivo:
                fecha, monto = linea.strip().split(',')
                self.ventas.append((datetime.strptime(fecha, "%Y-%m-%d"), float(monto)))
    def calcular_total_ventas(self):
        """
        Calcula el total de ventas del mes.
        Returns:
            float: El total de ventas.
        """
        return sum(venta[1] for venta in self.ventas)
    def calcular_promedio_diario(self):
        """
        Calcula el promedio de ventas diarias.
        Returns:
            float: El promedio de ventas diarias.
        """
        return self.calcular_total_ventas() / len(self.ventas)
    def identificar_dia_mayor_venta(self):
        """
        Identifica el día con mayores ventas.
        Returns:
            tuple: Una tupla con la fecha y el monto de la mayor venta.
        """
        return max(self.ventas, key=lambda x: x[1])
    def generar_informe(self):
        """
        Genera un informe con estadísticas básicas de ventas.
        Returns:
            str: Un informe con las estadísticas de ventas.
        """
        total_ventas = self.calcular_total_ventas()
        promedio_diario = self.calcular_promedio_diario()
        dia_mayor_venta, monto_mayor_venta = self.identificar_dia_mayor_venta()
        return f"""Informe de Ventas Mensuales:
Total de ventas: ${total_ventas:.2f}
Promedio de ventas diarias: ${promedio_diario:.2f}
Día con mayores ventas: {dia_mayor_venta.strftime('%Y-%m-%d')} (${monto_mayor_venta:.2f})"""
```

### **Tarea 3.** Implementar los tests unitarios.

**Paso 1.** En el archivo `test_analizador.py`, implementa los siguientes tests:

```python
import unittest
from analizador import AnalizadorVentas
import tempfile
import os
from datetime import datetime
class TestAnalizadorVentas(unittest.TestCase):
    def setUp(self):
        self.analizador = AnalizadorVentas()
        self.archivo_temporal = tempfile.NamedTemporaryFile(delete=False)
        with open(self.archivo_temporal.name, 'w') as f:
            f.write("fecha,monto\n")
            f.write("2024-01-01,1500.50\n")
            f.write("2024-01-02,2300.75\n")
            f.write("2024-01-03,1800.25\n")
    def tearDown(self):
        os.unlink(self.archivo_temporal.name)
    def test_cargar_datos(self):
        self.analizador.cargar_datos(self.archivo_temporal.name)
        self.assertEqual(len(self.analizador.ventas), 3)
    def test_calcular_total_ventas(self):
        self.analizador.cargar_datos(self.archivo_temporal.name)
        self.assertAlmostEqual(self.analizador.calcular_total_ventas(), 5601.50, places=2)
    def test_calcular_promedio_diario(self):
        self.analizador.cargar_datos(self.archivo_temporal.name)
        self.assertAlmostEqual(self.analizador.calcular_promedio_diario(), 1867.17, places=2)
    def test_identificar_dia_mayor_venta(self):
        self.analizador.cargar_datos(self.archivo_temporal.name)
        dia_mayor_venta, monto_mayor_venta = self.analizador.identificar_dia_mayor_venta()
        self.assertEqual(dia_mayor_venta, datetime(2024, 1, 2))
        self.assertAlmostEqual(monto_mayor_venta, 2300.75, places=2)
    def test_generar_informe(self):
        self.analizador.cargar_datos(self.archivo_temporal.name)
        informe = self.analizador.generar_informe()
        self.assertIn("Total de ventas: $5601.50", informe)
        self.assertIn("Promedio de ventas diarias: $1867.17", informe)
        self.assertIn("Día con mayores ventas: 2024-01-02 ($2300.75)", informe)
if __name__ == '__main__':
    unittest.main()
```

### **Tarea 4.** Crear el script principal.

**Paso 1.** En el archivo `main.py`, implementa el siguiente código:

```python
from analizador import AnalizadorVentas
def main():
    analizador = AnalizadorVentas()
    analizador.cargar_datos("ventas_mensuales.txt")
    informe = analizador.generar_informe()
    print(informe)
if __name__ == "__main__":
    main()
```

### **Tarea 5.** Crear el archivo de datos de ejemplo.

**Paso 1.** En el archivo `ventas_mensuales.txt`, ingresa algunos datos de ventas de ejemplo:

```
fecha,monto
2024-01-01,1500.50
2024-01-02,2300.75
2024-01-03,1800.25
2024-01-04,2100.00
2024-01-05,1950.75
```

### **Tarea 6.** Ejecutar los tests y la aplicación.

**Paso 1.** Para ejecutar los tests unitarios, usa el siguiente comando en la terminal:

   ```
   python -m unittest test_analizador.py
   ```
   
**Paso 2.** Para ejecutar la aplicación principal, usa:

   ```
   python main.py
   ```
   
## Resultado esperado: 

Este laboratorio te ha permitido practicar:

1. POO en Python: Creaste la clase `AnalizadorVentas`. <br>
2. Módulos en Python y Clases: Organizaste el código en módulos separados.<br>
3. Mantenibilidad y documentación: Uyilizaste docstrings y comentarios para documentar nuestro código.<br>
4. Testing Unitario: Implementaste tests unitarios para verificar el funcionamiento de la clase.

Además, has aplicado estos conceptos a un problema real de análisis de datos de ventas, lo cual es relevante para la ciencia de datos.


### [Índice](../README.md)

### [Práctica 2. Mejora de recomendaciones de productos](../Capítulo2/README.md)
