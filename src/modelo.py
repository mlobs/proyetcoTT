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
import json
import numpy as np
import joblib

columnas_ordenadas = [
    'cantidad_reportes_incidentes', 'hora_ingreso_al_sistema_1', 
    'hora_ingreso_area_simualcion_2', 'hora_ingreso_area_planificacion_3', 
    'hora_inicio_preparativos_4', 'hora_ingreso_area_tratamiento_5', 
    'hora_de_salida_paciente_6', 'verificacion_informacion_clinica', 
    'verificacion_ingreso_paciente_al_sistema', 'recepcion_ficha_medica', 
    'confirmacion_informe_ingreso', 'verificacion_identidad_paciente', 
    'verificacion_actividad_fuente', 'comprobacion_ubicacion_aplicadores', 
    'comprobacion_delimitacion_zona', 'comprobacion_indicacion_postura', 
    'validacion_informe_ingreso', 'comprobacion_repeticion_tratamiento', 
    'verificacion_prescripcion_dosis', 'finalizacion_analisis_curvas_dosis', 
    'verificacion_calculo_dosimetrico', 'finalizacion_segmentacion_oars', 
    'determinacion_lugar_colocacion', 'entrega_consentimiento_informado', 
    'verificacion_diaria_maquinaria', 'verificacion_consentimiento_informado', 
    'validacion_personal_asignado', 'cantidad_personal_asignado', 
    'validacion_tiempo_radiacion', 'validacion_estado_fuente', 
    'validacion_correcta_colocacion_aplicador', 'finalizacion_revision_tolrancia_fuente', 
    'entrega_medicamentos_necesarios', 'evaluacion_paciente', 
    'verificacion_controles_de_seguridad', 'verificacion_posicion_tubos_transferencia', 
    'verificar_informacion_plan', 'verificar_concordancia_tiempo_respecto_a_planificacion', 
    'verificacion_paciente', 'comprobacion_concordancia_tiemmpos_exposicion', 
    'registro_aplicadores_utilizados', 'registro_fotografico_aplicadores', 
    'finalizar_checklist_comprobacion', 'fecha_registro'
]

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

# Obtener la opción seleccionada para realizar predicciones
with open(r"data\opcion_seleccionada.json", "r") as file:
    data_json = json.load(file)
    opcion = str(data_json["opcion"])

    # Solicitar datos al usuario
    with open(r"data\datos.json","r") as file:
        datos_ingresados=json.load(file)

    entrada_df = pd.DataFrame([datos_ingresados])
    entrada_df = entrada_df[columnas_ordenadas]

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

    if opcion == "Linear Regression":
        modelo = joblib.load('LinearRegression.pkl')
    elif opcion == "Decision Tree":
        modelo = joblib.load('DecisionTree.pkl')
    elif opcion == "Random Forest":
        modelo = joblib.load('RandomForest.pkl')
    elif opcion == "Gradient Boosting":
        modelo = joblib.load('GradientBoosting.pkl')
    elif opcion == "Support Vector Regressor (SVR)":
        modelo = joblib.load('SVR.pkl')
    elif opcion == "K-Nearest Neighbors (KNN)":
        modelo = joblib.load('KNN.pkl')
    else:
        print("No se ha seleccionado una opción valida")

    # Realizar la predicción (asume que el modelo ya está entrenado)
    prediccion = modelo.predict(entrada_df)[0]
    resultados_prediccion = {
        "prediccion": float(prediccion),
    }

    with open(r'data\predict&proba.json', 'w') as file:
        json.dump(resultados_prediccion, file, indent=4)

    # Mostrar el resultado
    print(f"\nLa predicción del modelo '{opcion}' es: {prediccion} reclamos")