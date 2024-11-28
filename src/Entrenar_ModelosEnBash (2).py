import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.neural_network import MLPRegressor
#!pip install xgboost
#from xgboost import XGBRegressor
#!pip install lightgbm
from lightgbm import LGBMRegressor
import json
import numpy as np
import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = StandardScaler()  # Puedes usar MinMaxScaler() si prefieres escalar entre [0, 1]

opcion = ""
resultados = []

parametros = {
    "Linear Regression": {
        'fit_intercept': [True, False],
        'positive': [True, False]  # Usa este en lugar de parámetros obsoletos
    },
    "Decision Tree": {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['squared_error', 'friedman_mse']
    },
    "Random Forest": {
        'n_estimators': [100, 200, 500],
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [5, 10, 15],
        'min_samples_leaf': [2, 5, 10],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [False],
    },
    "Gradient Boosting": {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    },
    "SVR": {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.2]
    },
    "K-Nearest Neighbors (KNN)": {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    },
    "Regresion Bayesiana": {
        'alpha_1': [1e-6, 1e-5, 1e-4],  # Parámetro de regularización (prior alpha)
        'alpha_2': [1e-6, 1e-5, 1e-4],
        'lambda_1': [1e-6, 1e-5, 1e-4],  # Parámetro de regularización (prior lambda)
        'lambda_2': [1e-6, 1e-5, 1e-4],
    },
    "Redes Neuronales": {
        'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'max_iter': [500, 1000, 2000, 5000]
    },
    "XGBoost": {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],  # Porcentaje de columnas usadas por árbol
        'gamma': [0, 1, 5],  # Regularización L1
        'reg_alpha': [0, 0.1, 1],  # Regularización L2
        'reg_lambda': [0, 0.1, 1],
    },
    "LightGBM": {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [-1, 10, 20],
        'num_leaves': [31, 50, 100],  # Número de hojas máximo
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_samples': [10, 20, 30],  # Mínimos datos por hoja
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [0, 0.1, 1],
    },
}


#----------------------------------------------------------------------------------------------Tratamiento de datos i EDD --------------------------------------------------------
# Cargar el dataset
data = pd.read_csv(r'data\DatasetTT.csv')
# Preprocesamiento
for col in data.select_dtypes(include=['object']).columns:
    data[col] = data[col].str.replace(',', '.', regex=False)
# Conversión de las columnas necesarias
columnas_fecha_hora = [
    'hora_ingreso_al_sistema_1',
    'hora_ingreso_area_simualcion_2',
    'hora_ingreso_area_planificacion_3',
    'hora_inicio_preparativos_4',
    'hora_ingreso_area_tratamiento_5',
    'hora_de_salida_paciente_6',
    'fecha_registro'
]
# Convertir las columnas de horas a tipo datetime (por defecto con fecha arbitraria, solo se usa la hora)
data['hora_ingreso_al_sistema_1'] = pd.to_datetime(data['hora_ingreso_al_sistema_1'], format='%H:%M')
data['hora_ingreso_area_simualcion_2'] = pd.to_datetime(data['hora_ingreso_area_simualcion_2'], format='%H:%M')
data['hora_ingreso_area_planificacion_3'] = pd.to_datetime(data['hora_ingreso_area_planificacion_3'], format='%H:%M')
data['hora_inicio_preparativos_4'] = pd.to_datetime(data['hora_inicio_preparativos_4'], format='%H:%M')
data['hora_ingreso_area_tratamiento_5'] = pd.to_datetime(data['hora_ingreso_area_tratamiento_5'], format='%H:%M')
data['hora_de_salida_paciente_6'] = pd.to_datetime(data['hora_de_salida_paciente_6'], format='%H:%M')
# Calcular la diferencia entre cada par de columnas (en horas)
data['horas_entre_ingreso_y_simulacion'] = (data['hora_ingreso_area_simualcion_2'] - data['hora_ingreso_al_sistema_1']).dt.total_seconds() / 3600  # Diferencia en horas
data['horas_entre_simulacion_y_planificacion'] = (data['hora_ingreso_area_planificacion_3'] - data['hora_ingreso_area_simualcion_2']).dt.total_seconds() / 3600
data['horas_entre_planificacion_y_preparativos'] = (data['hora_inicio_preparativos_4'] - data['hora_ingreso_area_planificacion_3']).dt.total_seconds() / 3600
data['horas_entre_preparativos_y_tratamiento'] = (data['hora_ingreso_area_tratamiento_5'] - data['hora_inicio_preparativos_4']).dt.total_seconds() / 3600
data['horas_entre_tratamiento_y_salida'] = (data['hora_de_salida_paciente_6'] - data['hora_ingreso_area_tratamiento_5']).dt.total_seconds() / 3600
data['total_horas_en_atencion'] = (data['hora_de_salida_paciente_6'] - data['hora_ingreso_al_sistema_1']).dt.total_seconds() / 3600
data = data.drop(columns=columnas_fecha_hora, errors='ignore')
# Convertir las columnas numéricas de texto a float
data = data.apply(pd.to_numeric, errors='coerce')
# Manejar valores nulos
data = data.fillna(0)
# Separar las características (X) y la columna objetivo (y)
X = data.drop('cantidad_reclamos', axis=1)
y = data['cantidad_reclamos']
# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Modelos a probar

# Ajustar el escalador con los datos de entrenamiento y transformar los datos
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Solo transformamos el conjunto de prueba
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#st.write("Entrenar Modelos")
#opcion = st.selectbox("Elección del Modelo", ["","Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "Support Vector Regressor (SVR)", "K-Nearest Neighbors (KNN)"])

print("Entrenar Modelos")
print("1) Linear Regression", "\n2) Decision Tree", "\n3) Random Forest", "\n4) Gradient Boosting", "\n5) Support Vector Regressor (SVR) ", "\n6) K-Nearest Neighbors (KNN)","\n7) Regresion Bayesiana", "\n8) Redes Neuronales","\n9) XGBoost","\n10) LightGBM")
try:
    opcion=int(input("Elija un modelo: "))
    if opcion == 1:
        opcion = "Linear Regression"
    elif opcion == 2:
        opcion = "Decision Tree"
    elif opcion == 3:
        opcion = "Random Forest"
    elif opcion == 4:
        opcion = "Gradient Boosting"
    elif opcion == 5:
        opcion = "Support Vector Regressor (SVR)"
    elif opcion == 6:
        opcion = "K-Nearest Neighbors (KNN)"
    elif opcion == 7:
        opcion = "Regresion Bayesiana"
    elif opcion == 8:
        opcion = "Redes Neuronales"
    elif opcion == 9:
        opcion = "XGBoost"
    elif opcion == 10:
        opcion = "LightGBM"
    else:
        opcion = "Opción no válida"
except ValueError:
    opcion = "Entrada no válida"

print("Inicio Entrenamiento")


if opcion == "Linear Regression":
    modelo = LinearRegression()
elif opcion == "Decision Tree":
    modelo = DecisionTreeRegressor()#random_state=42),
elif opcion == "Random Forest":
    modelo = RandomForestRegressor()#random_state=42),
elif opcion == "Gradient Boosting":
    modelo = GradientBoostingRegressor()#random_state=42),
elif opcion == "Support Vector Regressor (SVR)":
    modelo = SVR()
elif opcion == "K-Nearest Neighbors (KNN)":
    modelo = KNeighborsRegressor()
elif opcion == "Regresion Bayesiana":
    modelo = BayesianRidge()
elif opcion == "Redes Neuronales":
    modelo = MLPRegressor()
#elif opcion == "XGBoost":
#    modelo = XGBRegressor()
elif opcion == "LightGBM":
    modelo = LGBMRegressor()


if opcion != "":
    # Entrenar y ajustar el modelo seleccionado
    modelo_seleccionado = modelo

    if opcion in parametros:
        # Aplicar GridSearchCV para ajustar el modelo
        grid_search = GridSearchCV(estimator=modelo, param_grid=parametros[opcion], cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        modelo_seleccionado = grid_search.best_estimator_
        print(f"Mejores parámetros para {opcion}: {grid_search.best_params_}")
    else:
        # Si no hay parámetros definidos, entrenar el modelo sin ajuste
        modelo_seleccionado.fit(X_train_scaled, y_train)
        print(f"Modelo {opcion} entrenado sin ajuste de parámetros.")

    if opcion == "Linear Regression":
        joblib.dump(modelo_seleccionado, 'data\\LinearRegression.pkl')
    elif opcion == "Decision Tree":
        joblib.dump(modelo_seleccionado, 'data\\DecisionTree.pkl')
    elif opcion == "Random Forest":
        joblib.dump(modelo_seleccionado, 'data\\RandomForest.pkl')
    elif opcion == "Gradient Boosting":
        joblib.dump(modelo_seleccionado, 'data\\GradientBoosting.pkl')
    elif opcion == "Support Vector Regressor (SVR)":
        joblib.dump(modelo_seleccionado, 'data\\SVR.pkl')
    elif opcion == "K-Nearest Neighbors (KNN)":
        joblib.dump(modelo_seleccionado, 'data\\KNN.pkl')
    elif opcion == "Regresion Bayesiana":
        joblib.dump(modelo_seleccionado, 'data\\BayesianRegression.pkl')
    elif opcion == "Redes Neuronales":
        joblib.dump(modelo_seleccionado, 'data\\NeuralNetwork.pkl')
    elif opcion == "XGBoost":
        joblib.dump(modelo_seleccionado, 'data\\XGBoost.pkl')
    elif opcion == "LightGBM":
        joblib.dump(modelo_seleccionado, 'data\\LightGBM.pkl')

 # Realizar predicciones
    y_pred = modelo_seleccionado.predict(X_test_scaled)

    # Calcular métricas de evaluación
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    # Guardar resultados
    resultados.append({
        "Modelo": opcion,
        "R²": r2,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
    })


    # 3. Histograma de residuos
    #st.subheader("Histograma de Residuos")
    # Supón que y_pred son las predicciones de tu modelo
    #residuos = y - y_pred
    #plt.scatter(y_pred, residuos)
   # plt.xlabel('Predicciones')
    #plt.ylabel('Residuos')
   # st.pyplot()  # Usar st.pyplot() para renderizar en Streamlit
    print("Fin Entrenamiento")
    print(f"Modelo Utilizado: {opcion}")
    print(f"**R²:** {r2}")
    print(f"**MSE:** {mse}")
    print(f"**RMSE:** {rmse}")
    print(f"**MAE:** {mae}")


    # Mostrar resultados de los modelos en un DataFrame
    #resultados_df = pd.DataFrame(resultados).sort_values(by="R²", ascending=False)
    #resultados_df.to_json(r'data\resultados_df.json', orient='records')
# Guardar métricas en un archivo de texto
    nombre_archivo = 'data\\metricas_modelo.txt'
    with open(nombre_archivo, 'a') as archivo:
        archivo.write(f" ")
        archivo.write(f"Modelo Utilizado: {opcion}\n")
        archivo.write(f"R²: {r2}\n")
        archivo.write(f"MSE: {mse}\n")
        archivo.write(f"RMSE: {rmse}\n")
        archivo.write(f"MAE: {mae}\n")

    print(f"Métricas guardadas en el archivo: {nombre_archivo}")
