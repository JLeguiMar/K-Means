import numpy as np #Calculos numéricos
import pandas as pd #Manipulación y análisis de datos
import matplotlib.pyplot as plt #permite crear visualizaciones
import seaborn as sns #Mejorar las visualizaciones 
import sklearn.cluster as cluster # Permite realizar los diferentes tipos de clustering(agrupamiento)
from sklearn.cluster import KMeans #Importa la clase KMeans, KMeans permite agrupar los datos en cluster
from sklearn.datasets import load_iris #Carga los datos de Iris

# Cargar el dataset Iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.head()

# Datos y etiquetas
x = iris.data # datos 
y = iris.target# especies 

# Crear y entrenar el modelo K-Means
kmeans = KMeans(n_clusters=3)
y_kmeans = kmeans.fit_predict(x)

# Visualizar los clusters y los centroides
plt.figure(figsize=(14, 7))
colormap = np.array(['red', 'lime', 'black'])
plt.scatter(x[:, 0], x[:, 1], c=colormap[y])  # Longitud vs. Ancho del sépalo
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=200, c='yellow', marker='X', edgecolor='k', label='Centroides')
plt.scatter(x[:, 0], x[:, 2], c=colormap[y])  # Longitud del sépalo vs. Longitud del pétalo
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 2], 
            s=200, c='yellow', marker='X', edgecolor='k')
plt.title('Clusters de Iris Dataset con k-Means')
plt.xlabel('Longitud de Sépalo')
plt.ylabel('Longitud de Pétalo')  
plt.legend() # Mostrar la leyenda para identificar los centroides
plt.show()

# Imprimir la precisión final del modelo en poercentaje
print(f'Exactitud del modelo: {accuracy*100:.2f}%')

# Predecir los valores para el conjunto de prueba
y_pred = clf.predict(x_test_flat)  # Genera las predicciones usando el modelo entrenado con los datos de prueba

# Calcular la matriz de confusión
cm = confusion_matrix(y_test, y_pred)  # Compara las etiquetas reales con las predichas y calcula la matriz de confusión

# Visualizar la matriz de confusión usando seaborn
plt.figure(figsize=(10, 7))  # Configura el tamaño de la figura para el gráfico de la matriz de confusión
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')  # Genera un mapa de calor de la matriz de confusión, con anotaciones de los valores
plt.title('Matriz de Confusión')  # Añade un título al gráfico
plt.xlabel('Etiqueta Predicha')  # Etiqueta para el eje X que indica las clases predichas por el modelo
plt.ylabel('Etiqueta Verdadera')  # Etiqueta para el eje Y que indica las clases reales o verdaderas
plt.show()  # Muestra el gráfico en la pantalla
