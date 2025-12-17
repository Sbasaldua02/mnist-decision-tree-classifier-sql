# Clasificación de Dígitos (MNIST) con Árboles de Decisión y Análisis SQL
# Integrantes: Santos Basaldúa, Solana Sabor y Emiliano Segovia

## Python | Pandas-Numpy | SQL |  Scikit-Learn (DecisionTreeClassifier, KFold) | Matplotlib, Seaborn

Este proyecto fue desarrollado para la materia **Laboratorio de Datos**. El objetivo principal fue implementar un modelo de clasificación supervisada capaz de distinguir dígitos manuscritos del dataset MNIST.

A diferencia de un enfoque tradicional, este trabajo pone un énfasis especial en la **interpretabilidad de los datos** (a nivel de píxeles) y en la **innovación técnica** al integrar consultas SQL para el análisis de métricas de validación cruzada.

### Análisis Exploratorio de Datos 
Realizamos un estudio profundo de la estructura de las imágenes de 28x28 píxeles para entender qué "miraba" el modelo:

* **Relevancia de Píxeles:** Identificamos que los píxeles con valores > 90 aportan la mayor información, mientras que los bordes son mayormente ruido (valor 0).
* **Mapas de Calor (Heatmaps):** Generamos el "dígito promedio" para cada clase.
    * *Hallazgo:* Los dígitos **5 y 6** presentaron la mayor similitud estructural, lo que explica las dificultades de clasificación entre ellos. Por el contrario, el 0 y el 1 son los más distinguibles.
      
      <img width="669" height="361" alt="image" src="https://github.com/user-attachments/assets/aaa6f97b-025b-4b4f-9480-7f6cd2a2d00c" />

* **Análisis de Variabilidad:** Calculamos la desviación estándar por píxel.
    * Se observó un "anillo" de baja desviación en el dígito 0 (color rojo en nuestros gráficos), indicando alta consistencia en la escritura de ese número.
      
      <img width="637" height="615" alt="image" src="https://github.com/user-attachments/assets/2e4d429d-27a0-4500-9137-eec24caaa49b" />


### 2. Selección de Datos
Para este experimento, trabajamos con un subconjunto de interés del dataset original, enfocándonos específicamente en las clases: **1, 2, 3, 7, y 9**.
* **Split:** 80% Desarrollo (Train) / 20% Validación (Hold-out).

### 3. Integración de SQL para Validación
En lugar de depender únicamente de Pandas, utilizamos **SQL** para procesar los resultados del *K-Fold Cross Validation* (5 folds). Esto nos permitió agregar métricas de manera dinámica:

''' sql
/* Ejemplo de la lógica utilizada */
SELECT 
    criterion, 
    alturas as profundidad, 
    AVG(train_accuracies) as avg_train, 
    AVG(test_accuracies) as avg_test
FROM resultados_folds
GROUP BY criterion, alturas
'''
### 4. Modelo y Experimentación

Para encontrar la configuración óptima, se realizó una búsqueda exhaustiva de hiperparámetros evaluando el desempeño del **DecisionTreeClassifier**:

* **Profundidad (`max_depth`):** Iteraciones en un rango de 1 a 20.
* **Criterio de División (`criterion`):** Comparación de rendimiento entre *Gini* y *Entropy*.
<img width="862" height="559" alt="image" src="https://github.com/user-attachments/assets/ed5b0442-e39a-4077-9f72-cd460c738d31" />


#### Resultados
Tras analizar las curvas de aprendizaje (*Bias* vs. *Variance*), seleccionamos el modelo final con los siguientes valores:

| Configuración | Valor Seleccionado |
| :--- | :--- |
| **Hiperparámetros** | `max_depth = 4` |
| **Criterio** | `Entropy` |
| **Performance (Validación)** | **~0.81 Accuracy** |

<img width="588" height="456" alt="image" src="https://github.com/user-attachments/assets/e6556539-7e66-4395-afc6-260a6d928373" />


> **Justificación (Bias-Variance):**
> Observamos que a partir de la **profundidad 4**, el modelo comienza a sufrir de *overfitting* (alta varianza): la curva de exactitud en *train* sigue subiendo, pero la de *validación* se estanca. Decidimos limitar la profundidad a 4 para mantener un equilibrio, evitando que el modelo "memorice" el ruido de los datos de entrenamiento.

