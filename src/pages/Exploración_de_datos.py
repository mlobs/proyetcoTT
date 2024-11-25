import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Cargar el dataset
data = pd.read_csv(r'data\DatasetTT.csv')

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

st.subheader("Exploración de Datos")

st.subheader("Cabeceras")
st.write(data.head())

st.subheader("Descripción de datos")
st.write(data.describe())

st.subheader("Análisis de datos")
cantidad_nulos = data.isnull().sum()#\
st.write("Cantidad de datos Nulos ",cantidad_nulos)

st.subheader("Dispersión de datos")

# Crear la figura y los ejes
st.subheader("Matriz de Correlación")
fig, ax = plt.subplots(figsize=(30, 20))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
st.pyplot(fig)