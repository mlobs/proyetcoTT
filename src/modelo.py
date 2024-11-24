import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import warnings
import json
import numpy as np


# Cargar el dataset
data = pd.read_csv(r'data\DatasetTT.csv')

warnings.filterwarnings('ignore')

# Preprocesamiento
warnings.filterwarnings('ignore')

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
modelos = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),#random_state=42),
    "Random Forest": RandomForestRegressor(),#random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(),#random_state=42),
    "Support Vector Regressor (SVR)": SVR(),
    "K-Nearest Neighbors (KNN)": KNeighborsRegressor()
}

# Parámetros para el ajuste
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
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
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
}

nombres_modelos = list(modelos.keys())
with open(r'data\modelos.json', 'w') as file:
    json.dump(nombres_modelos, file)

# Obtener la opción seleccionada para realizar predicciones
with open(r"data\opcion_seleccionada.json", "r") as file:
    data_json = json.load(file)
    opcion = int(data_json["opcion"])

if opcion != 0 :

    # Entrenar y ajustar el modelo seleccionado
    modelo_seleccionado = list(modelos.keys())[opcion - 1]
    if modelo_seleccionado in modelos.keys():
        modelo = modelos[modelo_seleccionado]
        
        if modelo_seleccionado in parametros:
            # Aplicar GridSearchCV para ajustar el modelo
            grid_search = GridSearchCV(estimator=modelo, param_grid=parametros[modelo_seleccionado], cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            modelos[modelo_seleccionado] = grid_search.best_estimator_
            print(f"Mejores parámetros para {modelo_seleccionado}: {grid_search.best_params_}")
        else:
            # Si no hay parámetros definidos, entrenar el modelo sin ajuste
            modelo.fit(X_train, y_train)
            modelos[modelo_seleccionado] = modelo
            print(f"Modelo {modelo_seleccionado} entrenado sin ajuste de parámetros.")
    else:
        print(f"El modelo {modelo_seleccionado} no está definido en la lista de modelos.")

    # Mostrar el desempeño de los modelos
    resultados = []
    if modelo_seleccionado in modelos.keys():
        # Obtener el modelo ya entrenado
        modelo = modelos[modelo_seleccionado]
        # Realizar predicciones
        y_pred = modelo.predict(X_test)
        # Calcular métricas de evaluación
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        # Guardar resultados
        resultados.append({
            "Modelo": modelo_seleccionado,
            "R²": r2,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae
        })
    else:
        print(f"El modelo {modelo_seleccionado} no está definido en la lista de modelos.")

    # Mostrar resultados de los modelos en un DataFrame
    resultados_df = pd.DataFrame(resultados).sort_values(by="R²", ascending=False)
    resultados_df.to_json(r'data\resultados_df.json', orient='records')

    nombre_modelo = list(modelos.keys())[opcion - 1]
    print(f"\nModelo seleccionado: {nombre_modelo}")

    with open(r"data\opcion_seleccionada.json", "r") as file:
        opcion = 0
#!--------------------------------------------------------------------------------------
    # Columnas simuladas (reemplaza con X.columns de tu dataset)
    columnas_ejemplo = [
        'cantidad_reportes_incidentes',
        'hora_ingreso_al_sistema_1',
        'hora_ingreso_area_simualcion_2',
        'hora_ingreso_area_planificacion_3',
        'hora_inicio_preparativos_4'
        'hora_ingreso_area_tratamiento_5',
        'hora_de_salida_paciente_6',
        'fecha_registro'
    ]
    with open(r'data\columnas_ejemplo.json', 'w') as file:
        json.dump(columnas_ejemplo, file)

    # Solicitar datos al usuario
    with open(r"data\datos.json","r") as file:
        datos_ingresados=json.load(file)

    entrada_df = pd.DataFrame([datos_ingresados])

    # Convertir las columnas de horas a tipo datetime (por defecto con fecha arbitraria, solo se usa la hora)
    entrada_df['hora_ingreso_al_sistema_1'] = pd.to_datetime(entrada_df['hora_ingreso_al_sistema_1'], format='%H:%M')
    entrada_df['hora_ingreso_area_simualcion_2'] = pd.to_datetime(entrada_df['hora_ingreso_area_simualcion_2'], format='%H:%M')
    entrada_df['hora_ingreso_area_planificacion_3'] = pd.to_datetime(entrada_df['hora_ingreso_area_planificacion_3'], format='%H:%M')
    entrada_df['hora_inicio_preparativos_4'] = pd.to_datetime(entrada_df['hora_inicio_preparativos_4'], format='%H:%M')
    entrada_df['hora_ingreso_area_tratamiento_5'] = pd.to_datetime(entrada_df['hora_ingreso_area_tratamiento_5'], format='%H:%M')
    entrada_df['hora_de_salida_paciente_6'] = pd.to_datetime(entrada_df['hora_de_salida_paciente_6'], format='%H:%M')

    # Calcular la diferencia entre cada par de columnas (en horas)
    entrada_df['horas_entre_ingreso_y_simulacion'] = (entrada_df['hora_ingreso_area_simualcion_2'] - entrada_df['hora_ingreso_al_sistema_1']).dt.total_seconds() / 3600  # Diferencia en horas
    entrada_df['horas_entre_simulacion_y_planificacion'] = (entrada_df['hora_ingreso_area_planificacion_3'] - entrada_df['hora_ingreso_area_simualcion_2']).dt.total_seconds() / 3600
    entrada_df['horas_entre_planificacion_y_preparativos'] = (entrada_df['hora_inicio_preparativos_4'] - entrada_df['hora_ingreso_area_planificacion_3']).dt.total_seconds() / 3600
    entrada_df['horas_entre_preparativos_y_tratamiento'] = (entrada_df['hora_ingreso_area_tratamiento_5'] - entrada_df['hora_inicio_preparativos_4']).dt.total_seconds() / 3600
    entrada_df['horas_entre_tratamiento_y_salida'] = (entrada_df['hora_de_salida_paciente_6'] - entrada_df['hora_ingreso_area_tratamiento_5']).dt.total_seconds() / 3600
    entrada_df['total_horas_en_atencion'] = (entrada_df['hora_de_salida_paciente_6'] - entrada_df['hora_ingreso_al_sistema_1']).dt.total_seconds() / 3600

    entrada_df = entrada_df.drop(columns = columnas_fecha_hora, errors='ignore')

    # Realizar la predicción (asume que el modelo ya está entrenado)
    prediccion = modelo.predict(entrada_df)[0]
    predentren = y_pred[0]
    print("(y_test.tolist(), y_pred.tolist()) == ",(y_test.tolist(), y_pred.tolist()))
    resultados_prediccion = {
        "prediccion entrenamiento": float(predentren),
        "prediccion": float(prediccion),
        "modelo": nombre_modelo,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mae": mae,
    
        "scatter": (y_test.tolist(), y_pred.tolist()),  # Para el gráfico de dispersión
        "residuos": (y_pred.tolist(), (y_test - y_pred).tolist()),  # Para el gráfico de residuos
        "histograma_residuos": (y_test - y_pred).tolist(),  # Para el histograma de residuos
    }

    with open(r'data\predict&proba.json', 'w') as file:
        json.dump(resultados_prediccion, file, indent=4)

    # Mostrar el resultado
    if prediccion > 0:
        print(f"Se predice que habrá {prediccion} reclamos")
        print(f"\nLa predicción del modelo '{nombre_modelo}' es: {prediccion} reclamos")
    else:
        print(f"Se predice que NO habrá reclamos")
        print(f"\nLa predicción del modelo '{nombre_modelo}' es: {prediccion} reclamos")