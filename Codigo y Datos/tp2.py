^^#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 10:39:20 2024

@author: 
"""

# script para cargar y plotear dígitos

#%%
# --- Librerías Estándar y de Sistema ---
import sqlite3

# --- Manipulación de Datos y SQL ---
import pandas as pd
import numpy as np
import pandasql as ps
from inline_sql import sql, sql_val

# --- Visualización ---
import matplotlib.pyplot as plt
import seaborn as sns

# --- Machine Learning (Scikit-Learn) ---
# Modelos y Árboles
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Selección de Modelos y Métricas
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
from sklearn.metrics import accuracy_score
#%% Cargar datos

# un array para las imágenes, otro para las etiquetas (por qué no lo ponen en el mismo array #$%@*)
data_imgs = np.load('C:/Users/Usuario/OneDrive/Escritorio/TP2/mnistc_images.npy')
data_chrs = np.load('C:/Users/Usuario/OneDrive/Escritorio/TP2/mnistc_labels.npy')[:,np.newaxis]


# mostrar forma del array:
# 1ra dimensión: cada una de las imágenes en el dataset
# 2da y 3ra dimensión: 28x28 píxeles de cada imagen
print(data_imgs.shape)
print(data_chrs.shape)



#%% Grafico imagen

# Elijo la imagen correspondiente a la letra que quiero graficar
n_digit = 18
image_array = data_imgs[n_digit,:,:,0]
image_label = data_chrs[n_digit]


# Ploteo el grafico
plt.figure(figsize=(10,8))
plt.imshow(image_array, cmap='gray')
plt.title('caracter: ' + str(image_label))
plt.axis('off')  
plt.show()
#%%
# 2.a
# Calcular promedio de cada píxel en todas las imágenes
pixel_promedio = (data_imgs.mean(axis=0)).T

pixel_promedio = np.squeeze(pixel_promedio)
# Visualizar un mapa de calor de las varianzas

sns.heatmap(pixel_promedio, cmap='Reds')
plt.title("promedio de los píxeles")
plt.xlabel("Pixel")  # Etiqueta para el eje X
plt.ylabel("Pixel")
plt.show()

#%% 
def calcular_promedio_por_numero(numeros, imagenes):
    # Crear un diccionario para almacenar las imágenes por número
    imagenes_por_numero = {i: [] for i in range(10)}
    
    # Agrupar las imágenes por el número correspondiente
    for num, img in zip(numeros, imagenes):
        imagenes_por_numero[int(num)].append(img)  # Asegurarse de que 'num' sea un entero
    
    # Crear un diccionario para almacenar los promedios de las imágenes por número
    promedios_por_numero = {}
    
    # Calcular el promedio de las imágenes para cada número
    for num in imagenes_por_numero:
        imagenes_lista = imagenes_por_numero[num]
        if imagenes_lista:  # Verificar que haya imágenes para el número
            imagenes_apiladas = np.stack(imagenes_lista)
            promedio = np.mean(imagenes_apiladas, axis=0)  # Promedio a lo largo de las imágenes
            promedios_por_numero[num] = promedio  # Usar el entero 'num' como clave
    
    return promedios_por_numero

# Llamada a la función para obtener el promedio de cada número
promedios_por_numero = calcular_promedio_por_numero(data_chrs, data_imgs)

def obtener_promedio_numero(promedios_por_numero, num):
    # Verificar si el promedio del número está calculado
    if num in promedios_por_numero:
        return promedios_por_numero[num]
    else:
        raise ValueError(f"No hay promedio calculado para el número {num}")
#%%
# 2.b
promedio_numero_0 = obtener_promedio_numero(promedios_por_numero, 0)
promedio_numero_1 = obtener_promedio_numero(promedios_por_numero, 1)
promedio_numero_5 = obtener_promedio_numero(promedios_por_numero, 5)
promedio_numero_6 = obtener_promedio_numero(promedios_por_numero, 6)


promedio_numero_0 = promedio_numero_0.reshape((28, 28))
promedio_numero_0 = np.rot90(promedio_numero_0, 0)

promedio_numero_1 = promedio_numero_1.reshape((28, 28))
promedio_numero_1 = np.rot90(promedio_numero_1, 0)

promedio_numero_5 = promedio_numero_5.reshape((28, 28))
promedio_numero_5 = np.rot90(promedio_numero_5, 0)

promedio_numero_6 = promedio_numero_6.reshape((28, 28))
promedio_numero_6 = np.rot90(promedio_numero_6, 0)
#%%
# 2.b

# Mostrar la imagen superpuesta con transparencia
plt.figure(figsize=(8, 6))
plt.imshow(promedio_numero_0, cmap="Reds",alpha=1, interpolation="nearest")
plt.imshow(promedio_numero_1, cmap="Greens",alpha=0.5, interpolation="nearest")
plt.colorbar()


plt.title('Diferenciando 0 y 1')
plt.show()
#%%
#2.b

plt.figure(figsize=(8, 6))
plt.imshow(promedio_numero_5, cmap="Reds",alpha=1, interpolation="nearest")
plt.imshow(promedio_numero_6, cmap="Greens",alpha=0.5, interpolation="nearest")
plt.colorbar()


plt.title('Diferenciando 5 y 6')
plt.show()

#%%
# 2.c
imagen_del_0 = []
for i in range (len(data_chrs)):
    if data_chrs[i] == 0:
      imagen_del_0.append(data_imgs[i])
                    
imagen_del_0_array = np.array(imagen_del_0)

imagen_del_0_array_ds = imagen_del_0_array.std(axis=0).reshape((28, 28)) #desviacion estandar de 0

plt.imshow(imagen_del_0_array_ds, cmap='hot')
plt.title(f"Variabilidad del dígito {0}")
plt.colorbar()
plt.show()
    
#%%
# 2.2.a

# Tomamos del dataset las imagenes de los digitos que nos interesan
imagenes_mis_numeros = []
data_mis_numeros= []
for i in range (len(data_chrs)):
    if data_chrs[i] == 1 or data_chrs[i] == 2 or data_chrs[i]==3 or data_chrs[i]==7 or data_chrs[i]==9 :
      imagenes_mis_numeros.append(data_imgs[i])
      data_mis_numeros.append(data_chrs[i])
imagen_mis_numeros_array = np.array(imagenes_mis_numeros)
data_mis_numeros_array = np.array(data_mis_numeros)
#%%
# 2.2.a
# aplanamos imagenes_mis_numeros_array en 2 dimensiones para poder usarlo en el arbol de decicion
x = imagen_mis_numeros_array.reshape(imagen_mis_numeros_array.shape[0], -1)
y = data_mis_numeros_array
#%%
# 2.2.a
# separamos entre datos de entrenamiento y evaluacion (hold_out)
x_train, x_eval, y_train, y_eval = train_test_split(x,y,random_state=1,test_size=0.2)
#%%
# 2.2.b
clf_info = tree.DecisionTreeClassifier(criterion = "entropy", max_depth=3)
clf_info = clf_info.fit(x_train, y_train)

# Generar los nombres de las características (por ejemplo, 'pixel_0', 'pixel_1', ..., 'pixel_783')
feature_names = ['pixel_{}'.format(i) for i in range(x.shape[1])]

# Generar los nombres de las clases (las etiquetas son 1, 2, 3, 7, y 9)
class_names = ['1', '2', '3', '7', '9']

# Visualizar el árbol de decisiones
plt.figure(figsize=[50, 35])
tree.plot_tree(clf_info, feature_names=feature_names, class_names=class_names, filled=True, rounded=True, fontsize=7)
plt.show()
#%%
#2.2.b
# Vemos la exactitud del arbol con los datos de entrenamiento
x_desarrollo, x_test, y_desarrollo, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=1)

# Listas para almacenar resultados
profundidades = range(1, 31)  # Probar profundidades de 1 a 10
exactitudes1 = []
exactitudes2 = []

# Entrenar y evaluar modelos para diferentes profundidades
for profundidad in profundidades:
    clf = DecisionTreeClassifier(max_depth=profundidad, criterion="entropy")
    clf.fit(x_desarrollo, y_desarrollo)
    Y_pred = clf.predict(x_test)
    exactitud = accuracy_score(y_test, Y_pred)
    exactitudes1.append(exactitud)
    
for profundidad in profundidades:
    clf = DecisionTreeClassifier(max_depth=profundidad, criterion="gini")
    clf.fit(x_desarrollo, y_desarrollo)
    Y_pred = clf.predict(x_test)
    exactitud = accuracy_score(y_test, Y_pred)
    exactitudes2.append(exactitud)
    
# Graficar los resultados
plt.figure(figsize=(10, 6))

plt.plot(profundidades, exactitudes1, marker='o', color = 'blue', label= 'entropy' )
plt.plot(profundidades, exactitudes2, marker = 'o', color = 'orange', label = 'gini')

plt.title('Exactitud del Árbol de Decisión según la Profundidad')
plt.xlabel('Profundidad del Árbol')
plt.ylabel('Exactitud')
plt.xticks(profundidades)
plt.grid()
plt.legend()
plt.show()

#%%
# 2.2.c
#Usando los datos de desarrollo que separamos anteriormente, realizamos una evaluacion del modelo de arbol para clasificacion probando con distintas profundidades y criterios
Desarrollo_X = x_train
Desarrollo_Y = y_train

#pasamos los array a dataframe
Desarrollo_X = pd.DataFrame(Desarrollo_X)
Desarrollo_Y = pd.DataFrame(Desarrollo_Y)


# Parámetros del modelo
rango_alturas = range(1, 21)
criteria = ['gini', 'entropy']
clf_info = []
i= 0
nsplits = 5
kf = KFold(n_splits=nsplits)

# Loop para cada criterio
for altura in rango_alturas:
    for criterion in criteria:
        # Crear un modelo con la configuración actual de criterio y profundidad
        model = DecisionTreeClassifier(max_depth=altura, criterion=criterion,random_state=16)
        
        #para ver que fold es
        i= 0

        # Cross-validation manual en cada fold
        for train_index, val_index in kf.split(Desarrollo_X):
            # Dividir los datos de entrenamiento y validación en cada fold
            X_train_fold, X_val_fold = Desarrollo_X.iloc[train_index], Desarrollo_X.iloc[val_index]
            Y_train_fold, Y_val_fold = Desarrollo_Y.iloc[train_index], Desarrollo_Y.iloc[val_index]

            # Entrenar el modelo en el fold actual
            model.fit(X_train_fold, Y_train_fold)

            # Calcular precisión en el conjunto de entrenamiento
            Y_train_pred = model.predict(X_train_fold)
            train_score = accuracy_score(Y_train_fold, Y_train_pred)

            # Calcular precisión en el conjunto de validación
            Y_val_pred = model.predict(X_val_fold)
            val_score = accuracy_score(Y_val_fold, Y_val_pred)
            
            #para el fold
            i+=1

            # Guardar la información en el diccionario `clf_info`
            clf_info.append({
               'criterion': criterion,
               'alturas': altura,
               'fold': i,
               'train_accuracies': train_score,
               'test_accuracies': val_score
           })
        
df_clf_info = pd.DataFrame(clf_info)        
#%%
# Consulta SQL
consultaSQL = """
    SELECT criterion, alturas, AVG(train_accuracies) AS promedio_exactitud_train, AVG(test_accuracies) AS promedio_exactitud_test
    FROM df_clf_info
    GROUP BY criterion, alturas
    ORDER BY alturas DESC;
"""

# Pasa el DataFrame explícitamente en un diccionario
evaluacion_arboles = ps.sqldf(consultaSQL, locals())
#%%
consultaSQL = """
    SELECT *
    FROM evaluacion_arboles
    WHERE criterion = 'entropy'
"""

evaluacion_entropy = ps.sqldf(consultaSQL, locals())

consultaSQL = """
    SELECT *
    FROM evaluacion_arboles
    WHERE criterion = 'gini'
"""

evaluacion_gini = ps.sqldf(consultaSQL, locals())
#%%
fig, ax = plt.subplots()
 
plt.rcParams['font.family'] = 'sans-serif'           
 
ax.plot('alturas', 'promedio_exactitud_train', data= evaluacion_entropy,
        marker='.',  linestyle='-', linewidth=0.9, color= 'red', label='exactitud entropy con train')

ax.plot('alturas', 'promedio_exactitud_test', data=evaluacion_entropy,
        marker='.', linestyle='--', linewidth=0.9, color= 'orange', label='exactitud entropy con test')

ax.plot('alturas', 'promedio_exactitud_train', data=evaluacion_gini,
        marker='.', linestyle='-',  linewidth=0.9, color= 'blue', label='exactitud gini con train')    

ax.plot('alturas', 'promedio_exactitud_test', data=evaluacion_gini,
        marker='.', linestyle='--', linewidth=0.9, color= 'skyblue', label='exactitud gini con test')   

ax.set_title('Exactitud con Entropia y Gini', fontsize = 15)
ax.set_xlabel('Profundidad del arbol', fontsize = 14)
ax.set_ylabel('Promedio de Exactitud', fontsize = 14)
ax.set_xlim(0,21)
ax.set_xticks(range(0,21,1))
ax.legend()
plt.show()


#%%
#2.2.d
# Para este ejercicio vamos a utilizar la variable 'y_eval' que es el conjunto de datos hold-out que representa el 20% de los datos totales que van a ser usados para la validacion 
# como el criterio de entropia en profundidad 4 nos devolvia mas exactitud que con gini decidimos este criterio
clasificador = DecisionTreeClassifier(max_depth=4, criterion='entropy',random_state=16)
clasificador.fit(x_train, y_train)

y_predict = clasificador.predict(x_eval)       

print("Exactitud del modelo:", metrics.accuracy_score(y_eval, y_predict))
matriz_confusion= metrics.confusion_matrix(y_eval, y_predict) 
print(matriz_confusion)
matriz_confusion = pd.DataFrame(matriz_confusion)