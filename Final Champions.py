#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Generar datos sintéticos
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'team1_goals': np.random.randint(0, 5, n_samples),
    'team2_goals': np.random.randint(0, 5, n_samples),
    'possession': np.random.randint(30, 70, n_samples),
    'shots_on_target': np.random.randint(0, 15, n_samples),
    'yellow_cards': np.random.randint(0, 5, n_samples),
    'red_cards': np.random.randint(0, 2, n_samples)
})

# Crear la columna de resultado del partido
def determine_result(row):
    if row['team1_goals'] > row['team2_goals']:
        return 'Victoria'
    elif row['team1_goals'] < row['team2_goals']:
        return 'Derrota'
    else:
        return 'Empate'

data['match_result'] = data.apply(determine_result, axis=1)

# Transformar la columna 'match_result' a valores numéricos
le = LabelEncoder()
data['match_result'] = le.fit_transform(data['match_result'])

# Seleccionar características y etiquetas
features = data[['team1_goals', 'team2_goals', 'possession', 'shots_on_target', 'yellow_cards', 'red_cards']]
labels = data['match_result']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Definir el modelo
model = RandomForestClassifier(random_state=42)

# Usar GridSearchCV para encontrar los mejores hiperparámetros
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Mejor modelo encontrado
best_model = grid_search.best_estimator_

# Evaluar el modelo
y_pred = best_model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Hacer la predicción para el partido específico
match_data = pd.DataFrame({
    'team1_goals': [2],
    'team2_goals': [1],
    'possession': [55],
    'shots_on_target': [7],
    'yellow_cards': [3],
    'red_cards': [0]
})

prediction = best_model.predict(match_data)
predicted_result = le.inverse_transform(prediction)
print(f'Predicción del partido: {predicted_result[0]}')


# In[9]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Generar datos sintéticos
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'team1_goals': np.random.randint(0, 5, n_samples),
    'team2_goals': np.random.randint(0, 5, n_samples),
    'possession': np.random.randint(30, 70, n_samples),
    'shots_on_target': np.random.randint(0, 15, n_samples),
    'yellow_cards': np.random.randint(0, 5, n_samples),
    'red_cards': np.random.randint(0, 2, n_samples)
})

# Crear la columna de resultado del partido
def determine_result(row):
    if row['team1_goals'] > row['team2_goals']:
        return 'Victoria'
    elif row['team1_goals'] < row['team2_goals']:
        return 'Derrota'
    else:
        return 'Empate'

data['match_result'] = data.apply(determine_result, axis=1)

# Transformar la columna 'match_result' a valores numéricos
le = LabelEncoder()
data['match_result'] = le.fit_transform(data['match_result'])

# Seleccionar características y etiquetas
features = data[['team1_goals', 'team2_goals', 'possession', 'shots_on_target', 'yellow_cards', 'red_cards']]
labels = data['match_result']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Definir el modelo
model = RandomForestClassifier(random_state=42)

# Usar GridSearchCV para encontrar los mejores hiperparámetros
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Mejor modelo encontrado
best_model = grid_search.best_estimator_

# Evaluar el modelo
y_pred = best_model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Graficar la matriz de confusión
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, display_labels=le.classes_, cmap=plt.cm.Blues)
plt.title('Matriz de Confusión')
plt.savefig('confusion_matrix.png')
plt.show()

# Hacer la predicción para el partido específico
match_data = pd.DataFrame({
    'team1_goals': [2],
    'team2_goals': [1],
    'possession': [55],
    'shots_on_target': [7],
    'yellow_cards': [3],
    'red_cards': [0]
})

prediction = best_model.predict(match_data)
predicted_result = le.inverse_transform(prediction)
print(f'Predicción del partido: {predicted_result[0]}')

# Obtener probabilidades de resultado
probabilities = best_model.predict_proba(match_data)[0]
labels = le.classes_

# Crear el gráfico circular de probabilidades de resultado
fig, ax = plt.subplots()
ax.pie(probabilities, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
ax.axis('equal')  # Para asegurar que el círculo sea circular
plt.title('Real Madrid vs Borussia Dortmund')
plt.savefig('probabilidades_resultado.png')
plt.show()

# Gráfico de Barras Comparativo para Estadísticas del Partido
stats = {
    'Posesión (%)': [55, 45],  # Real Madrid, Borussia Dortmund
    'Disparos a puerta': [7, 5],
    'Tarjetas amarillas': [3, 2],
    'Tarjetas rojas': [0, 1]
}

labels = ['Real Madrid', 'Borussia Dortmund']
pos = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(12, 8))

bar_width = 0.35
opacity = 0.8

# Asegurar que cada estadística tiene dos valores
for i, (key, values) in enumerate(stats.items()):
    plt.bar(pos + i * bar_width, values, bar_width, alpha=opacity, label=key)

plt.xlabel('Equipos')
plt.ylabel('Valores')
plt.title('Comparación de Estadísticas: Real Madrid vs Borussia Dortmund')
plt.xticks(pos + bar_width, labels)
plt.legend()

plt.tight_layout()
plt.savefig('comparacion_estadisticas.png')
plt.show()

