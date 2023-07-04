#%%
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


# %%
#Leo el excel
df = pd.read_csv('train.csv')

# %%
#Me fijo si los id estan duplicados
# Verificar si la columna 'Nombre' tiene valores duplicados
duplicados = df['id'].duplicated()

# Imprimir los registros que tienen valores duplicados
print("La cantidad de registros duplicaodos son:",duplicados)

#No tiene duplicados entonces se puede eliminar

# %%
#Elimino el indice y el id
df = df.drop('id', axis=1)
df = df.drop(df.columns[0], axis=1)

#%%
#Creo dos nuevas variables que marquen si hubo atraso en el aterrizage o en el despegue

df['Arrival Delay Indicator'] = df['Arrival Delay in Minutes'].apply(lambda x: 1 if x > 0 else 0)
df['Departure Delay Indicator'] = df['Departure Delay in Minutes'].apply(lambda x: 1 if x > 0 else 0)

#%%
#Reemplazo la variable de arrival delay por la resta entre departure delay y arrival

df['Arrival Delay in Minutes']=df['Departure Delay in Minutes'] - df['Arrival Delay in Minutes']

print(df['Arrival Delay in Minutes'])

df['Atraso/Adelanto Indicator'] = df['Arrival Delay in Minutes'].apply(lambda x: 1 if x >= 0 else 0)


#%%
#Ahora voy a crear grupos de edades diferentes

edades_bins = [0, 12, 19, 40, 65, 100]
edades_labels = ['0', '2', '4', '6','8']

# Dividir la variable de edad en grupos utilizando pd.cut()
df['Grupo Edad'] = pd.cut(df['Age'], bins=edades_bins, labels=edades_labels)

print(df['Grupo Edad'])

# %%
#ENCODING POR FRECUENCIA
#Paso las variables categoricas a numericas
#Paso la variable Gender, costumer type, Type of travel a una dummy
df['Gender'] = pd.get_dummies(df['Gender'], prefix='Gender', drop_first=True).astype(int)
df['Customer Type'] = pd.get_dummies(df['Customer Type'], prefix='Customer Type', drop_first=True).astype(int)
df['Type of Travel'] = pd.get_dummies(df['Type of Travel'], prefix='Type of Travel', drop_first=True).astype(int)

#Para class hago un encoding ordinal

# Definir el diccionario de codificación ordinal
encoding_dict = {'Eco': 1, 'Business': 3, 'Eco Plus': 2}
# Aplicar el encoding ordinal a la columna 'Class'
df['Class'] = df['Class'].map(encoding_dict)

# Definir el diccionario de codificación ordinal
encoding_dict2 = {'satisfied': 0, 'neutral or dissatisfied':1}
# Aplicar el encoding ordinal a la columna 'Nivel'
df['satisfaction'] = df['satisfaction'].map(encoding_dict2)


#%%

# Variables a incluir en el promedio ponderado
variables_a_ponderar = ['Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness']

# Calcular las correlaciones por grupo de edad
correlations_grouped = df.groupby('Grupo Edad')[variables_a_ponderar + ['satisfaction']].corr()['satisfaction']

# Calcular los pesos por grupo de edad basados en las correlaciones
weights_grouped = correlations_grouped.drop('satisfaction', level=1)  # Eliminar la suma de las correlaciones dentro de cada grupo
weights_grouped = weights_grouped.div(weights_grouped.groupby('Grupo Edad').transform('sum'))  # Dividir por la suma total de correlaciones por grupo

# Calcular el promedio ponderado por grupo de edad
df['Promedio Ponderado'] = 0.0  # Crear una columna vacía para almacenar el promedio ponderado
for group, group_df in df.groupby('Grupo Edad'):
    weights = weights_grouped.loc[group]  # Obtener los pesos correspondientes al grupo de edad
    variables = group_df[variables_a_ponderar]  # Obtener las variables a ponderar del grupo de edad
    total_weight = weights.sum()  # Suma total de los pesos
    weighted_average = (variables * (weights / total_weight)).sum(axis=1)  # Calcular el promedio ponderado
    df.loc[group_df.index, 'Promedio Ponderado'] = weighted_average  # Asignar el promedio ponderado al dataframe original

# %%

# Verificar qué variables contienen NaN
variables_con_nan = df.columns[df.isnull().any()].tolist()

# Imprimir las variables con NaN
print("Variables con NaN:", variables_con_nan)

# Imputar los valores faltantes de Arrival Delay in Minutes utilizando los valores de Departure Delay in Minutes
df['Arrival Delay in Minutes'] = df['Departure Delay in Minutes'].fillna(df['Arrival Delay in Minutes'])

#%%
#Analizo la variable de distancia
import matplotlib.pyplot as plt
y=df['Flight Distance']
x = [0] * len(y)

# Graficar la variable como una línea con puntos de tamaño reducido
plt.plot(x, y, marker='o', linestyle='', markersize=3)

# Etiqueta del eje y
plt.ylabel('Variable')

# Título del gráfico
plt.title('Gráfico de línea con puntos achicados')

# Mostrar el gráfico
plt.show()


# %%
#Ahora voy a crear grupos con kmeans
from sklearn.cluster import KMeans
import numpy as np

# Crear un objeto KMeans con el número deseado de grupos
kmeans = KMeans(n_clusters=3)

# Ajustar el modelo a los datos
kmeans.fit(df['Flight Distance'].values.reshape(-1, 1))

# Obtener las etiquetas de los grupos
labels = kmeans.labels_
df['TipoDestino'] = labels
print(df['TipoDestino'] )
# Obtener los centroides de los grupos
centroids = kmeans.cluster_centers_

# Imprimir las etiquetas de los grupos y los centroides
print("Etiquetas de los grupos:", labels)
print("Centroides:", centroids)

# %%
# Crear un array de colores para asignar a cada tipo de destino
colores = np.array(['red', 'blue', 'orange', 'purple'])

# Graficar la variable como una línea con puntos de tamaño reducido y colores según el tipo de destino
plt.scatter(x, y, c=colores[df['TipoDestino']], marker='o', s=3)

# Etiqueta del eje y
plt.ylabel('Variable')

# Título del gráfico
plt.title('Gráfico de línea con puntos achicados (por tipo de destino)')

# Mostrar el gráfico
plt.show()

# %%
#Ahora voy a agregar una columna segun el tipo de pasajero que es

from sklearn.preprocessing import StandardScaler

# Seleccionar las variables relevantes para la clasificación
variables = ['Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class']

# Crear un dataframe con las variables seleccionadas
df_variables = df[variables]

# Estandarizar los datos
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df_variables)

# Crear un objeto KMeans con el número deseado de grupos
kmeans = KMeans(n_clusters=3)

# Ajustar el modelo a los datos estandarizados
kmeans.fit(data_scaled)

# Obtener las etiquetas de los grupos
labels = kmeans.labels_

#Agrego al dataframe el grupo 
df['TipoPasajero'] = labels

# Obtener los centroides de los grupos
centroids = kmeans.cluster_centers_

# Analizar los resultados
print("Etiquetas de los grupos:", labels)
print("Centroides:", centroids)

#%%
#Ahora grafico
from sklearn.decomposition import PCA

# Reducción de dimensionalidad con PCA
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

# Graficar los puntos asignados a cada grupo
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels, cmap='viridis')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Grupos de personas')
plt.colorbar(label='Grupo')
plt.show()

#%%
import seaborn as sns

# Crear un DataFrame con los centroides
df_centroids = pd.DataFrame(centroids, columns=variables)

# Crear el heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df_centroids, cmap='YlGnBu', annot=True, fmt=".2f", cbar=True)
plt.title('Heatmap de los Centroides')
plt.xlabel('Variables')
plt.ylabel('Grupos/Clústeres')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()




#%%
# Crear objeto PCA y ajustarlo a los datos
pca = PCA(n_components=2)
pca.fit(data_scaled)

# Obtener los vectores de carga para cada componente principal
loadings = pca.components_

# Crear un DataFrame para almacenar los vectores de carga
loadings_df = pd.DataFrame(loadings.T, columns=['Componente Principal 1', 'Componente Principal 2'], index=variables)

# Imprimir los vectores de carga
print("Vectores de Carga:")
print(loadings_df)



# %%
#Mapa de correlaciones
import seaborn as sns
import matplotlib.pyplot as plt

# Calcular la matriz de correlación
correlation_matrix = df.corr()

# Crear el mapa de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, cmap='coolwarm', linewidths=0.5, annot=False)

# Añadir estilo al mapa de correlación
plt.title('Mapa de Correlación', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# Mostrar el gráfico
plt.show()



# %%
#Arranco a probar modelos

dfp=df.drop('satisfaction',axis=1)

#Defino X e y
X = dfp
y = df['satisfaction']


#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV

#%%

# Definir los hiperparámetros a ajustar
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
}

# Crear el modelo de Random Forest
rf_model = RandomForestClassifier()

# Realizar el Grid Search
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

# Obtener los mejores hiperparámetros y el mejor modelo
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(best_params,"los mejores ",best_model)

#%%
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

# Pruebo Random Forest
# Crear el clasificador Random Forest
rf_classifier = RandomForestClassifier(min_samples_split=5, n_estimators=300)

# Dividir los datos en conjunto de entrenamiento y conjunto de prueba
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el clasificador con los datos de entrenamiento
rf_classifier.fit(X_train_rf, y_train_rf)

# Realizar predicciones en los datos de prueba
y_pred_rf = rf_classifier.predict(X_test_rf)

print(y_pred_rf)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test_rf, y_pred_rf)
print("Precisión del modelo: {:.2f}".format(accuracy))

# Calcular la matriz de confusión
confusion_matrix_rf = confusion_matrix(y_test_rf, y_pred_rf)
print("Matriz de confusión:")
print(confusion_matrix_rf)

# Calcular el reporte de clasificación
classification_report = classification_report(y_test_rf, y_pred_rf)
print("Reporte de clasificación:")
print(classification_report)


# Calcular las probabilidades de predicción para el cálculo de la curva ROC
y_pred_proba_rf = rf_classifier.predict_proba(X_test_rf)[:, 1]

# Calcular el valor del AUC-ROC
auc_roc_rf = roc_auc_score(y_test_rf, y_pred_proba_rf)
print("AUC-ROC: {:.2f}".format(auc_roc_rf ))

# Calcular los valores de la curva ROC
fpr, tpr, thresholds = roc_curve(y_test_rf, y_pred_proba_rf)

# Graficar la curva ROC
plt.plot(fpr, tpr, label='Curva ROC (AUC = {:.2f})'.format(auc_roc_rf))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.8, 1.1])  # Ajustar el límite superior para una mejor visualización
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()

# Calcular el AUC para cada umbral
table_rf  = pd.DataFrame({'Umbral': thresholds, 'AUC': [auc_roc_rf] * len(thresholds)})
table_rf  = table_rf [['Umbral', 'AUC']]
print("Tabla de AUC por umbral:")
print(table_rf )


# Obtener el nombre de las variables
feature_names_rf = X.columns

# Crear un DataFrame con la importancia de las variables
importance_rf = rf_classifier.feature_importances_
importance_df_rf = pd.DataFrame({'Variable': feature_names_rf, 'Importancia': importance_rf})
importance_df_rf = importance_df_rf.sort_values('Importancia', ascending=False)

# Mostrar la importancia de las variables
print("La importancia de las variables es:", importance_rf)

#%%
# Realizar validación cruzada con precisión RANDOM FOREST
# Realizar validación cruzada con precisión
accuracy_scores_rf = cross_val_score(rf_classifier, X, y, cv=5, scoring='accuracy')

# Calcular la precisión media y desviación estándar
mean_accuracy_rf = accuracy_scores_rf.mean()
std_accuracy_rf = accuracy_scores_rf.std()

# Imprimir los resultados
print("Precisión promedio: {:.2f}".format(mean_accuracy_rf))
print("Desviación estándar de precisión: {:.2f}".format(std_accuracy_rf))

#%%
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt


#PRUEBO CON REGRESIÓN LOGÍSTICA
X_train_rl, X_test_rl, y_train_rl, y_test_rl = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Escalado de variables
sc_X = StandardScaler()
X_train_rl = sc_X.fit_transform(X_train_rl)
X_test_rl = sc_X.transform(X_test_rl)

# Ajustar el modelo de Regresión Logística en el Conjunto de Entrenamiento
classifier_rl = LogisticRegression()
classifier_rl.fit(X_train_rl, y_train_rl)


# Predicción de los resultados con el Conjunto de Testing
y_pred_rl  = classifier_rl.predict(X_test_rl)


# Calcular la precisión del modelo
accuracy = accuracy_score(y_test_rl, y_pred_rl)
print("Precisión del modelo: {:.2f}".format(accuracy))

# Calcular la matriz de confusión
from sklearn.metrics import confusion_matrix
confusion_matrix_rl = confusion_matrix(y_test_rl, y_pred_rl)
print("Matriz de confusión:")
print(confusion_matrix_rl)

# Calcular el reporte de clasificación
classification_report_rl = classification_report(y_test_rl, y_pred_rl)
print("Reporte de clasificación:")
print(classification_report_rl)


# Calcular las probabilidades de predicción para el cálculo de la curva ROC
y_pred_proba_rl = classifier_rl.predict_proba(X_test_rl)[:, 1]

# Calcular el valor del AUC-ROC
auc_roc_rl = roc_auc_score(y_test_rl, y_pred_proba_rl)
print("AUC-ROC: {:.2f}".format(auc_roc_rl))

# Calcular los valores de la curva ROC
fpr, tpr, thresholds = roc_curve(y_test_rl, y_pred_proba_rl)

# Graficar la curva ROC
plt.plot(fpr, tpr, label='Curva ROC (AUC = {:.2f})'.format(auc_roc_rl))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.2])  # Ajustar el límite superior para una mejor visualización
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()

# Calcular el AUC para cada umbral
table_rl = pd.DataFrame({'Umbral': thresholds, 'AUC': [auc_roc_rl] * len(thresholds)})
table_rl = table_rl[['Umbral', 'AUC']]
print("Tabla de AUC por umbral:")
print(table_rl)

# Obtener los coeficientes de las variables
coefficients_rl = classifier_rl.coef_[0]

# Obtener el nombre de las variables
feature_names_rl = X.columns

# Crear un DataFrame con la importancia de las variables
importance_rl = pd.DataFrame({'Variable': feature_names_rl, 'Importancia': coefficients_rl})
importance_rl = importance_rl.sort_values('Importancia', ascending=False)

# Mostrar la importancia de las variables
print("La importancia de las variables es:", importance_rl)

#%%
# Realizar validación cruzada con precisión REGRESION LOGISTICA
accuracy_scores_rl = cross_val_score(classifier_rl, X, y, cv=5, scoring='accuracy')

# Calcular la precisión media y desviación estándar
mean_accuracy_rl = accuracy_scores_rl.mean()
std_accuracy_rl = accuracy_scores_rl.std()

# Imprimir los resultados
print("Precisión promedio: {:.2f}".format(mean_accuracy_rl))
print("Desviación estándar de precisión: {:.2f}".format(std_accuracy_rl))

# %%
#Pruebo con Suport Vector Machines

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

# Crear el clasificador SVM
svm_classifier = SVC(kernel = 'linear', probability = True)

# Dividir los datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test= sc.transform(X_test)

clf = SVC()
clf.fit(X_train, y_train)

"""
clf=SVC(C=500 , gamma=0.001 ,kernel='rbf')
clf.fit(X_train_scaled , y_train)
"""

# Entrenar el clasificador con los datos de entrenamiento
svm_classifier.fit(X_train, y_train)

# Realizar predicciones en los datos de prueba
y_pred = svm_classifier.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo: {:.2f}".format(accuracy))

# Calcular la matriz de confusión
confusion_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de confusión:")
print(confusion_matrix)

# Calcular el reporte de clasificación
classification_report = classification_report(y_test, y_pred)
print("Reporte de clasificación:")
print(classification_report)

# Calcular las probabilidades de predicción para el cálculo de la curva ROC
y_pred_proba = svm_classifier.predict_proba(X_test)[:, 1]

# Calcular el valor del AUC-ROC
auc_roc = roc_auc_score(y_test, y_pred_proba)
print("AUC-ROC: {:.2f}".format(auc_roc))

# Calcular los valores de la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Graficar la curva ROC
plt.plot(fpr, tpr, label='Curva ROC (AUC = {:.2f})'.format(auc_roc))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.2])  # Ajustar el límite superior para una mejor visualización
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()

# Calcular el AUC para cada umbral
table = pd.DataFrame({'Umbral': thresholds, 'AUC': [auc_roc] * len(thresholds)})
table = table[['Umbral', 'AUC']]
print("Tabla de AUC por umbral:")
print(table)

# Obtener el nombre de las variables
feature_names = X.columns

# Crear un DataFrame con la importancia de las variables
importance = svm_classifier.coef_[0]
importance_df = pd.DataFrame({'Variable': feature_names, 'Importancia': importance})
importance_df = importance_df.sort_values('Importancia', ascending=False)

# Mostrar la importancia de las variables
print("La importancia de las variables es:", importance)

     
# %%

# Definir las características categóricas
cat_features = ['Grupo Edad']  # Asegúrate de incluir otras columnas categóricas si las tienes

# Definir los hiperparámetros a ajustar
param_grid = {
    'iterations': [100, 200, 300],
    'learning_rate': [0.1, 0.01, 0.001],
    'depth': [4, 6, 8],
}

# Crear el modelo de CatBoost Classifier con las características categóricas especificadas
catboost_model = CatBoostClassifier(cat_features=cat_features)

# Realizar el Grid Search
grid_search = GridSearchCV(estimator=catboost_model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

# Obtener los mejores hiperparámetros y el mejor modelo
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(best_params,best_model)

#%%
#Pruebo con Catboost
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
# Crear una lista de características categóricas
cat_features = ['Grupo Edad']  # Agrega 'Grupo Edad' a la lista de características categóricas

# Crear un modelo CatBoost
catboost_classifier = CatBoostClassifier(cat_features=cat_features, iterations=200,depth=8,learning_rate=0.1)


# Split the data into training and test sets
X_train_cb, X_test_cb, y_train_cb, y_test_cb = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the classifier with the training data
catboost_classifier.fit(X_train_cb, y_train_cb)

# Make predictions on the test data
y_pred_cb = catboost_classifier.predict(X_test_cb)

# Calculate the model accuracy
accuracy_cb = accuracy_score(y_test_cb, y_pred_cb)
print("Model Accuracy: {:.2f}".format(accuracy_cb))

# Calculate the confusion matrix
confusion_matrix_cb = confusion_matrix(y_test_cb, y_pred_cb)
print("Confusion Matrix:")
print(confusion_matrix_cb)

# Calculate the predicted probabilities for ROC curve calculation
y_pred_proba_cb = catboost_classifier.predict_proba(X_test_cb)[:, 1]

# Calculate the AUC-ROC value
auc_roc_cb = roc_auc_score(y_test_cb, y_pred_proba_cb)
print("AUC-ROC: {:.2f}".format(auc_roc_cb))

# Calculate the values for the ROC curve
fpr, tpr, thresholds = roc_curve(y_test_cb, y_pred_proba_cb)

# Plot the ROC curve
plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(auc_roc_cb))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.2])  # Adjust the upper limit for better visualization
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Calculate the AUC for each threshold
table_cb = pd.DataFrame({'Threshold': thresholds, 'AUC': [auc_roc_cb] * len(thresholds)})
table_cb = table_cb[['Threshold', 'AUC']]
print("AUC by Threshold Table:")
print(table_cb)

# Get the feature names
feature_names_cb = X.columns

# Create a DataFrame with feature importances
importance_cb = catboost_classifier.feature_importances_
importance_df_cb = pd.DataFrame({'Variable': feature_names_cb, 'Importance': importance_cb})
importance_df_cb = importance_df_cb.sort_values('Importance', ascending=False)

# Show the feature importances
print("Feature Importances:")
#print(importance_df_cb)

# Calculate the classification report
classification_report_cb = classification_report(y_test_cb, y_pred_cb)
print("Classification Report:")
print(classification_report_cb)


print(confusion_matrix_cb)
print(classification_report_cb)

#%%
# Realizar validación cruzada con precisión CATBOOST
# Realizar validación cruzada con precisión
accuracy_scores_CB = cross_val_score(catboost_classifier, X, y, cv=5, scoring='accuracy')

# Calcular la precisión media y desviación estándar
mean_accuracy_CB = accuracy_scores_CB.mean()
std_accuracy_CB = accuracy_scores_CB.std()

# Imprimir los resultados
print("Precisión promedio: {:.2f}".format(mean_accuracy_CB))
print("Desviación estándar de precisión: {:.2f}".format(std_accuracy_CB))

# %%

#Pruebo con Naive Bayes
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

# Create the Naive Bayes classifier
nb_classifier = GaussianNB()

# Split the data into training and test sets
X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the classifier with the training data
nb_classifier.fit(X_train_nb, y_train_nb)

# Make predictions on the test data
y_pred_nb = nb_classifier.predict(X_test_nb)

# Calculate the model accuracy
accuracy_nb = accuracy_score(y_test_nb, y_pred_nb)
print("Model Accuracy: {:.2f}".format(accuracy_nb))

# Calculate the confusion matrix
confusion_matrix_nb = confusion_matrix(y_test_nb, y_pred_nb)
print("Confusion Matrix:")
print(confusion_matrix_nb)

# Calculate the classification report
classification_report_nb = classification_report(y_test_nb, y_pred_nb)
print("Classification Report:")
print(classification_report_nb)

# Calculate the predicted probabilities for ROC curve calculation
y_pred_proba_nb = nb_classifier.predict_proba(X_test_nb)[:, 1]

# Calculate the AUC-ROC value
auc_roc_nb = roc_auc_score(y_test_nb, y_pred_proba_nb)
print("AUC-ROC: {:.2f}".format(auc_roc_nb))

# Calculate the values for the ROC curve
fpr, tpr, thresholds = roc_curve(y_test_nb, y_pred_proba_nb)

# Plot the ROC curve
plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(auc_roc_nb))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.2])  # Adjust the upper limit for better visualization
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.ylim([0.0, 0.5]) 
plt.show()

# Calculate the AUC for each threshold
table_nb = pd.DataFrame({'Threshold': thresholds, 'AUC': [auc_roc_nb] * len(thresholds)})
table_nb = table_nb[['Threshold', 'AUC']]
print("AUC by Threshold Table:")
print(table_nb)

# Get the feature names
feature_names_nb = X.columns

# Naive Bayes doesn't have a built-in feature importance measure, so we'll skip that as well.

print(confusion_matrix_nb)

#%%
# Realizar validación cruzada con precisión NAIVE BAYES
# Realizar validación cruzada con precisión
accuracy_scores_nb = cross_val_score(nb_classifier, X, y, cv=5, scoring='accuracy')

# Calcular la precisión media y desviación estándar
mean_accuracy_nb = accuracy_scores_nb.mean()
std_accuracy_nb= accuracy_scores_nb.std()

# Imprimir los resultados
print("Precisión promedio: {:.2f}".format(mean_accuracy_nb))
print("Desviación estándar de precisión: {:.2f}".format(std_accuracy_nb))

# %%
#Pruebo con Nearest Neighbours Create the k-NN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Split the data into training and test sets
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the classifier with the training data
knn_classifier.fit(X_train_knn, y_train_knn)

# Make predictions on the test data
y_pred_knn = knn_classifier.predict(X_test_knn)

# Calculate the model accuracy
accuracy_knn = accuracy_score(y_test_knn, y_pred_knn)
print("Model Accuracy: {:.2f}".format(accuracy_knn))

# Calculate the confusion matrix
confusion_matrix_knn = confusion_matrix(y_test_knn, y_pred_knn)
print("Confusion Matrix:")
print(confusion_matrix_knn)

# Calculate the classification report
classification_report_knn = classification_report(y_test_knn, y_pred_knn)
print("Classification Report:")
print(classification_report_knn)

# k-NN doesn't provide predicted probabilities, so we can skip the ROC curve and AUC-ROC calculation.

# Get the feature names
feature_names_knn = X.columns

# k-NN doesn't have a built-in feature importance measure, so we'll skip that as well.

print(confusion_matrix_knn)

#%%
# Realizar validación cruzada con precisión NEAREST NEIGHBOURS
# Realizar validación cruzada con precisión
accuracy_scores_nn = cross_val_score(knn_classifier, X, y, cv=5, scoring='accuracy')

# Calcular la precisión media y desviación estándar
mean_accuracy_nn = accuracy_scores_nn.mean()
std_accuracy_nn= accuracy_scores_nn.std()

# Imprimir los resultados
print("Precisión promedio: {:.2f}".format(mean_accuracy_nn))
print("Desviación estándar de precisión: {:.2f}".format(std_accuracy_nn))


#%%

# Definir los hiperparámetros a ajustar
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
}

# Crear el modelo de Extra Trees Classifier
et_model = ExtraTreesClassifier()

# Realizar el Grid Search
grid_search = GridSearchCV(estimator=et_model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

# Obtener los mejores hiperparámetros y el mejor modelo
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(best_params,"los mejores ",best_model)

# %%
#Pruebo con Extra Trees
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

et_classifier = ExtraTreesClassifier(min_samples_split=5, n_estimators=300)

# Split the data into training and test sets
X_train_et, X_test_et, y_train_et, y_test_et = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the classifier with the training data
et_classifier.fit(X_train_et, y_train_et)

# Make predictions on the test data
y_pred_et = et_classifier.predict(X_test_et)

# Calculate the model accuracy
accuracy_et = accuracy_score(y_test_et, y_pred_et)
print("Model Accuracy: {:.2f}".format(accuracy_et))

# Calculate the confusion matrix
confusion_matrix_et = confusion_matrix(y_test_et, y_pred_et)
print("Confusion Matrix:")
print(confusion_matrix_et)

# Calculate the classification report
classification_report_et = classification_report(y_test_et, y_pred_et)
print("Classification Report:")
print(classification_report_et)

# Calculate the predicted probabilities for ROC curve calculation
y_pred_proba_et = et_classifier.predict_proba(X_test_et)[:, 1]

# Calculate the AUC-ROC value
auc_roc_et = roc_auc_score(y_test_et, y_pred_proba_et)
print("AUC-ROC: {:.2f}".format(auc_roc_et))

# Calculate the values for the ROC curve
fpr, tpr, thresholds = roc_curve(y_test_et, y_pred_proba_et)

# Plot the ROC curve
plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(auc_roc_et))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.2])  # Adjust the upper limit for better visualization
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Calculate the AUC for each threshold
table_et = pd.DataFrame({'Threshold': thresholds, 'AUC': [auc_roc_et] * len(thresholds)})
table_et = table_et[['Threshold', 'AUC']]
print("AUC by Threshold Table:")
print(table_et)

# Get the feature names
feature_names_et = X.columns

# Create a DataFrame with feature importances
importance_et = et_classifier.feature_importances_
importance_df_et = pd.DataFrame({'Variable': feature_names_et, 'Importance': importance_et})
importance_df_et = importance_df_et.sort_values('Importance', ascending=False)

# Show the feature importances
print("Feature Importances:")
print(importance_df_et)

print(confusion_matrix_et)

#%%
# Realizar validación cruzada con precisión EXTRA TREES
# Realizar validación cruzada con precisión
accuracy_scores_et = cross_val_score(et_classifier, X, y, cv=5, scoring='accuracy')

# Calcular la precisión media y desviación estándar
mean_accuracy_et = accuracy_scores_et.mean()
std_accuracy_et= accuracy_scores_et.std()

# Imprimir los resultados
print("Precisión promedio: {:.2f}".format(mean_accuracy_et))
print("Desviación estándar de precisión: {:.2f}".format(std_accuracy_et))
# %%
