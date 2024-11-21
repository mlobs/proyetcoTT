import streamlit as st
import pandas as pd
import json
import subprocess
from datetime import date, datetime

#!           streamlit run c:/Users/Gustavo/Desktop/TrabajoDeTitulo/prototipoModeloDeCalidad/STstreamlit.py
prediccion=0
valiPredi=False
st.set_page_config(page_title="Mi aplicación", layout="wide", initial_sidebar_state="expanded")

st.write("Frontend con Streamlit está corriendo...")

#*  Mostrar el desempeño de los modelos
st.write("""*Mostrar el desempeño de los modelos*""")
mod = pd.read_json(r'data\resultados_df.json')
st.write(mod)

#*  Mostrando los Modelos en pantalla
with st.container():
    with open(r'data\modelos.json', 'r') as file:
        nombres_modelos = json.load(file)
    st.write("Modelos disponibles:")
    for i, nombre in enumerate(nombres_modelos):
        st.write(f"{i+1}.{nombre}")

#*  Eleccion el Modelo
opcion = st.selectbox("Elección del Modelo", [0, 1, 2, 3, 4, 5], index=0)
with open(r"data\opcion_seleccionada.json", "w") as file:
    json.dump({"opcion": opcion}, file)



#? Esta linea de codigo entrega las columnas de ejemplo con las que se esta trabajando al momento
with open(r'data\columnas_ejemplo.json', 'r') as file:
    columnasUtilizadas = json.load(file)
st.write(columnasUtilizadas)


if opcion:
    for i, nombre in enumerate(nombres_modelos):
        if opcion-1 == i: 
            modelo_elegido=nombre
            st.write(f"Se ha elegido el modelo:{modelo_elegido}")

    #! Valores para prediccion
    st.write("Ingrese los datos necesarios para realizar la predicción: ")
    
    
# Función para procesar cada entrada
    def procesar_entrada(columnas):
        # Diccionario para almacenar las entradas
        entrada = {}

        # Entrada de fecha  ----------------------------------------------------------------------------------------------------------------------------
        entrada["fecha_registro"] = st.date_input("Ingrese la fecha del tratamiento (formato DD/MM/AAAA):",date.today())
        # Convertir la fecha a DD/MM/YYYY
        entrada["fecha_registro"] = entrada["fecha_registro"].strftime("%d-%m-%Y")

        # Entrada de horas  ----------------------------------------------------------------------------------------------------------------------------
        entrada["hora_ingreso_al_sistema_1"] = st.time_input("Seleccione la hora de ingreso al sistema (formato HH:MM):", key="hora_ingreso_al_sistema_1")
        entrada["hora_ingreso_area_simualcion_2"] = st.time_input("Seleccione la hora de ingreso al área de simulación (formato HH:MM):", key="hora_ingreso_area_simualcion_2")
        entrada["hora_ingreso_area_planificacion_3"] = st.time_input("Seleccione la hora de inicio de la planificación (formato HH:MM):", key="hora_ingreso_area_planificacion_3")
        entrada["hora_inicio_preparativos_4"] = st.time_input("Seleccione la hora a la que se iniciaron los preparativos (formato HH:MM):", key="hora_inicio_preparativos_4")
        entrada["hora_ingreso_area_tratamiento_5"] = st.time_input("Seleccione la hora de inicio del tratamiento (formato HH:MM):", key="hora_ingreso_area_tratamiento_5")
        entrada["hora_de_salida_paciente_6"] = st.time_input("Seleccione la hora a la que se retiró el paciente (formato HH:MM):", key="hora_de_salida_paciente_6")
        # Convertir las horas a formato HH:MM
        entrada["hora_ingreso_al_sistema_1"] = entrada["hora_ingreso_al_sistema_1"].strftime("%H:%M")
        entrada["hora_ingreso_area_simualcion_2"] = entrada["hora_ingreso_area_simualcion_2"].strftime("%H:%M")
        entrada["hora_ingreso_area_planificacion_3"] = entrada["hora_ingreso_area_planificacion_3"].strftime("%H:%M")
        entrada["hora_inicio_preparativos_4"] = entrada["hora_inicio_preparativos_4"].strftime("%H:%M")
        entrada["hora_ingreso_area_tratamiento_5"] = entrada["hora_ingreso_area_tratamiento_5"].strftime("%H:%M")
        entrada["hora_de_salida_paciente_6"] = entrada["hora_de_salida_paciente_6"].strftime("%H:%M")


        # Entrada de cantidad de incidentes  --------------------------------------------------------------------------------------------------------------
        entrada["cantidad_reportes_incidentes"] = st.number_input("Ingrese la cantidad de incidentes reportados",step=1,min_value=0,key="cantidad_reportes_incidentes")

        # Entrada de verificaciones  -------------------------------------------------------------------------------------------------------------------------------  
        entrada["verificacion__informacion_clinica"] = 1 if st.toggle("¿Se verificó la información clínica necesaria?", key="verificacion__informacion_clinica") else 0
        entrada["verificacion_ingreso_paciente_al_sistema"] = 1 if st.toggle("¿Se ingresó al paciente en el sistema?", key="verificacion_ingreso_paciente_al_sistema") else 0
        entrada["recepcion_ficha_medica"] = 1 if st.toggle("¿Se recepcionó la ficha médica?", key="recepcion_ficha_medica") else 0
        entrada["confirmacion_informe_ingreso"] = 1 if st.toggle("¿Se confirmó la recepción del informe de ingreso?", key="confirmacion_informe_ingreso") else 0
        entrada["verificacion_identidad_paciente"] = 1 if st.toggle("¿Se verificó la identidad del paciente?", key="verificacion_identidad_paciente") else 0
        entrada["verificacion_actividad_fuente"] = 1 if st.toggle("¿Se verificó la actividad de la fuente?", key="verificacion_actividad_fuente") else 0
        entrada["comprobacion_ubicacion_aplicadores"] = 1 if st.toggle("¿Se comprobó la correcta ubicación de los aplicadores?", key="comprobacion_ubicacion_aplicadores") else 0
        entrada["comprobacion_delimitacion_zona"] = 1 if st.toggle("¿Finalizó la delimitación de la zona de aplicación?", key="comprobacion_delimitacion_zona") else 0
        entrada["comprobacion_indicacion_postura"] = 1 if st.toggle("¿Se le dieron al paciente las indicaciones de postura?", key="comprobacion_indicacion_postura") else 0
        entrada["validacion_informe_ingreso"] = 1 if st.toggle("¿Se comprobó la información del informe de ingreso?", key="validacion_informe_ingreso") else 0

        entrada["comprobacion_repeticion_tratamiento"] = 1 if st.toggle("Finalización Análisis de Curvas de dosis", key="comprobacion_repeticion_tratamiento") else 0
        entrada["verificacion_prescripcion_dosis"] = 1 if st.toggle("Finalización Análisis de Curvas de dosis", key="verificacion_prescripcion_dosis") else 0
        entrada["finalizacion_analisis_curvas_dosis"] = 1 if st.toggle("Finalización Análisis de Curvas de dosis", key="finalizacion_analisis_curvas_dosis") else 0
        entrada["verificacion_calculo_dosimetrico"] = 1 if st.toggle("Finalización Análisis de Curvas de dosis", key="verificacion_calculo_dosimetrico") else 0
        entrada["finalizacion_segmentacion_oars"] = 1 if st.toggle("Finalización Análisis de Curvas de dosis", key="finalizacion_segmentacion_oars") else 0
        entrada["determinacion_lugar_colocación"] = 1 if st.toggle("Finalización Análisis de Curvas de dosis", key="determinacion_lugar_colocación") else 0
        entrada["entrega_consentimiento_informado"] = 1 if st.toggle("Finalización Análisis de Curvas de dosis", key="entrega_consentimiento_informado") else 0
        entrada["verificacion_diaria_maquinaria"] = 1 if st.toggle("Finalización Análisis de Curvas de dosis", key="verificacion_diaria_maquinaria") else 0
        entrada["verificacion_consentimiento_informado"] = 1 if st.toggle("Finalización Análisis de Curvas de dosis", key="verificacion_consentimiento_informado") else 0
        entrada["validacion_personal_asignado"] = 1 if st.toggle("Finalización Análisis de Curvas de dosis", key="validacion_personal_asignado") else 0
        entrada["cantidad_personal_asignado"] = 1 if st.toggle("Finalización Análisis de Curvas de dosis", key="cantidad_personal_asignado") else 0
        entrada["validacion_tiempo_radiacion"] = 1 if st.toggle("Finalización Análisis de Curvas de dosis", key="validacion_tiempo_radiacion") else 0
        entrada["validacion_estado_fuente"] = 1 if st.toggle("Finalización Análisis de Curvas de dosis", key="validacion_estado_fuente") else 0
        entrada["validacion_correcta_colocacion_aplicador"] = 1 if st.toggle("Finalización Análisis de Curvas de dosis", key="validacion_correcta_colocacion_aplicador") else 0
        entrada["finalizacion_revision_tolrancia_fuente"] = 1 if st.toggle("Finalización Análisis de Curvas de dosis", key="finalizacion_revision_tolrancia_fuente") else 0
        entrada["entrega_medicamentos_necesarios"] = 1 if st.toggle("Finalización Análisis de Curvas de dosis", key="entrega_medicamentos_necesarios") else 0
        entrada["evaluacion_paciente"] = 1 if st.toggle("Finalización Análisis de Curvas de dosis", key="evaluacion_paciente") else 0
        entrada["verificacion_controles_calidad"] = 1 if st.toggle("Finalización Análisis de Curvas de dosis", key="verificacion_controles_calidad") else 0
        entrada["verificacion_posicion_tubos_transferencia"] = 1 if st.toggle("Finalización Análisis de Curvas de dosis", key="verificacion_posicion_tubos_transferencia") else 0
        entrada["verificar_informacion_plan"] = 1 if st.toggle("Finalización Análisis de Curvas de dosis", key="verificar_informacion_plan") else 0
        entrada["verificar_concordancia_tiempo_respecto_a_planificacion"] = 1 if st.toggle("Finalización Análisis de Curvas de dosis", key="verificar_concordancia_tiempo_respecto_a_planificacion") else 0
        entrada["verificacion_paciente"] = 1 if st.toggle("Finalización Análisis de Curvas de dosis", key="verificacion_paciente") else 0
        entrada["comprobacion_concordancia_tiemmpos_exposicion"] = 1 if st.toggle("Finalización Análisis de Curvas de dosis", key="comprobacion_concordancia_tiemmpos_exposicion") else 0
        entrada["registro_aplicadores_utilizados"] = 1 if st.toggle("Finalización Análisis de Curvas de dosis", key="registro_aplicadores_utilizados") else 0
        entrada["registro_fotografico_aplicadores"] = 1 if st.toggle("Finalización Análisis de Curvas de dosis", key="registro_fotografico_aplicadores") else 0
        entrada["finalizar_checklist_comprobacion"] = 1 if st.toggle("Finalización Análisis de Curvas de dosis", key="finalizar_checklist_comprobacion") else 0

        return entrada  # Devolver el diccionario con los valores ingresados

    # Llamar a la función y obtener los valores ingresados
    entrada = procesar_entrada(columnasUtilizadas)

# Botón para guardar los datos y ejecutar el backend
    if st.button("Enviar"):
            valiPredi=True
            # Guardar los datos en un archivo JSON
            with open(r"data\datos.json", "w") as file:
                json.dump(entrada, file)

            # Iniciar el backend
            backend_process = subprocess.run(["python", "src\modelo.py"])
            # Cargar la predicción desde el archivo
            with open(r'data\predict&proba.json', "r") as file:
                prediccion = json.load(file)

    if valiPredi is True:
    # Mostrar la predicción en un expander
        with st.expander("Predicción"):
            if prediccion is not None:
                st.write(f"La prediccion del modelo {modelo_elegido}")
                st.write(f" es de un total de {prediccion} reclamos")
            else:
                st.write("La predicción no está disponible aún.")