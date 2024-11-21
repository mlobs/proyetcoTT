# Librerías necesarias
import pandas as pd
#from IPython.display import display
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
#import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import warnings
import json


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
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Support Vector Regressor (SVR)": SVR(),
    "K-Nearest Neighbors (KNN)": KNeighborsRegressor()
}
nombres_modelos = list(modelos.keys())
with open(r'data\modelos.json', 'w') as file:
    json.dump(nombres_modelos, file)


# Entrenar todos los modelos y almacenarlos
modelos_entrenados = {}
for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    modelos_entrenados[nombre] = modelo

# Mostrar el desempeño de los modelos
resultados = []
for nombre, modelo in modelos_entrenados.items():
    y_pred = modelo.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    resultados.append({"Modelo": nombre, "R²": r2, "RMSE": rmse})
resultados_df = pd.DataFrame(resultados).sort_values(by="R²", ascending=False)
resultados_df.to_json(r'data\resultados_df.json', orient='records')


with open(r"data\opcion_seleccionada.json", "r") as file:
    data_json = json.load(file)
    opcion = int(data_json["opcion"])

if opcion != 0 :

    modelo_seleccionado = list(modelos_entrenados.values())[opcion - 1]
    nombre_modelo = list(modelos_entrenados.keys())[opcion - 1]
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
    prediccion = modelo_seleccionado.predict(entrada_df)[0]
    # probabilidad = modelo_seleccionado.predict_proba(entrada_df)[0][1]

    with open(r'data\predict&proba.json', 'w') as file:
        json.dump(prediccion, file)

    # Mostrar el resultado
    if prediccion >= 1:
        print(f"Se predice que habrá {int(prediccion)} reclamos con un r2 de {r2}.")
        print(f"\nLa predicción del modelo '{nombre_modelo}' es: {prediccion} reclamos")
    else:
        print(f"Se predice que NO habrá reclamos con un r2 de {r2}%.")
        print(f"\nLa predicción del modelo '{nombre_modelo}' es: {prediccion} reclamos")