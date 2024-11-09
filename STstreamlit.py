import streamlit as st
import pandas as pd
import json

#data = pd.read_json(r'C:\Users\Gustavo\Desktop\TrabajoDeTitulo\prototipoModeloDeCalidad\data.json', orient='records')
#st.write(data)

# Mostrar el desempeño de los modelos
st.write("""*Mostrar el desempeño de los modelos*""")
mod = pd.read_json(r'C:\\Users\\Gustavo\\Desktop\\TrabajoDeTitulo\\prototipoModeloDeCalidad\\resultados_df.json')
st.write(mod)

#Mostrando los Modelos en pantalla
with open(r'C:\Users\Gustavo\Desktop\TrabajoDeTitulo\prototipoModeloDeCalidad\modelos.json', 'r') as file:
    nombres_modelos = json.load(file)
st.write("Modelos disponibles:")
st.write(nombres_modelos)