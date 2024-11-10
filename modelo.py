# Librerías necesarias
import pandas as pd
from IPython.display import display
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import warnings
import json
import os


# Vaciar los archivos JSON al iniciar el programa
def inicializar_json(archivo_json):
    with open(archivo_json, 'w') as file:
        json.dump(0, file)  # Guardar un diccionario vacío

# Lista de archivos JSON a inicializar
archivos_a_inicializar = ['valor.json', 'resultados_df.json','modelos.json']

for archivo in archivos_a_inicializar:
    if archivos_a_inicializar == 'opcion_seleccionada.json':
        with open("opcion_seleccionada.json", "w") as file:
            json.dump({"opcion": 0}, file)
    inicializar_json(archivo)

# Cargar el dataset
data = pd.read_csv(r'DatasetTT.csv')

# Preprocesamiento
warnings.filterwarnings('ignore')

# Preprocesamiento
for col in data.select_dtypes(include=['object']).columns:
    data[col] = data[col].str.replace(',', '.', regex=False)

# Conversión de las columnas necesarias
columnas_fecha_hora = [
    'Hora de Ingreso del paciente en el sistema',
    'Hora de ingreso en el area simualcion',
    'Hora de ingreso en el area planificacion preparativos',
    'Hora de ingreso en el area de Tratamiento',
    'Fecha de Realización'
]
data = data.drop(columns=columnas_fecha_hora, errors='ignore')

# Convertir las columnas numéricas de texto a float
data = data.apply(pd.to_numeric, errors='coerce')

# Manejar valores nulos
data = data.fillna(0)

# Separar las características (X) y la columna objetivo (y)
X = data.drop('Cantidad de Reclamos', axis=1)
y = data['Cantidad de Reclamos']

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
with open(r'modelos.json', 'w') as file:
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
resultados_df.to_json(r'resultados_df.json', orient='records')
#print(resultados_df)

# Seleccionar el modelo para la prueba interactiva

#print("\nSeleccione un modelo para usar en la predicción:")
#for i, modelo in enumerate(modelos_entrenados.keys()):
#    print(f"{i + 1}. {modelo}")
# Cargar la opción desde el archivo


with open("opcion_seleccionada.json", "r") as file:
    data = json.load(file)
    opcion = int(data["opcion"])

if opcion != 0 :

    modelo_seleccionado = list(modelos_entrenados.values())[opcion - 1]
    nombre_modelo = list(modelos_entrenados.keys())[opcion - 1]
    print(f"\nModelo seleccionado: {nombre_modelo}")
    

    # Columnas simuladas (reemplaza con X.columns de tu dataset)
    columnas_ejemplo = [
        "Cantidad reportes de Incidentes",
        "Hora de Ingreso del paciente en el sistema",
        "Hora de ingreso en el area simualcion",
        "Hora de ingreso en el area planificacion preparativos",
        "Hora de ingreso en el area de Tratamiento",
        "Fecha de Realización"
    ]
    with open(r'columnas_ejemplo.json', 'w') as file:
        json.dump(columnas_ejemplo, file)

    # Solicitar datos al usuario
    with open('valor.json',"r") as file:
        datos_ingresados=json.load(file)


    """if "Hora" in columna:  # Detectar columnas relacionadas con horas
        valor = input(f"Ingrese el valor para {columna} (formato HH:MM, default=00:00): ")
        try:
            entrada[columna] = datetime.strptime(valor, '%H:%M').time() if valor.strip() else time(0, 0)
        except ValueError:
            print(f"Formato inválido para {columna}. Se usará el valor por defecto: 00:00.")
            entrada[columna] = time(0, 0)
    elif "Fecha" in columna:  # Detectar columnas relacionadas con fechas
        valor = input(f"Ingrese el valor para {columna} (formato DD/MM/AAAA, default=01/01/2000): ")
        try:
            entrada[columna] = datetime.strptime(valor, '%d/%m/%Y').date() if valor.strip() else date(2000, 1, 1)
        except ValueError:
            print(f"Formato inválido para {columna}. Se usará el valor por defecto: 01/01/2000.")
            entrada[columna] = date(2000, 1, 1)"""
"""
    # Convertir tiempos y fechas a cadenas para el modelo
    for columna in datos_ingresados:
        if isinstance(datos_ingresados[columna], datetime.time):
            datos_ingresados[columna] = datos_ingresados[columna].strftime('%H:%M:%S')
        elif isinstance(datos_ingresados[columna], datetime.date):
            datos_ingresados[columna] = datos_ingresados[columna].strftime('%Y-%m-%d')

    # Convertir entrada a DataFrame
    entrada_df = pd.DataFrame([datos_ingresados])

    # Realizar la predicción (asume que el modelo ya está entrenado)
    prediccion = modelo_seleccionado.predict(entrada_df)[0]
    probabilidad = modelo_seleccionado.predict_proba(entrada_df)[0][1]

    # Mostrar el resultado
    if prediccion == 1:
        print(f"Se predice que habrá reclamos con una probabilidad del {probabilidad*100:.2f}%.")
        print(f"\nLa predicción del modelo '{nombre_modelo}' es: {prediccion}")
    else:
        print(f"Se predice que NO habrá reclamos con una probabilidad del {(1-probabilidad)*100:.2f}%.")
        print(f"\nLa predicción del modelo '{nombre_modelo}' es: {prediccion}")"""