import streamlit as st
import json
import subprocess
import matplotlib.pyplot as plt
import seaborn as snspop
from datetime import date

#!           streamlit run c:\\Users\\Gustavo\\Desktop\\TrabajoDeTitulo\\proyetcoTT\\src\\STstreamlit.py

prediccion=0
global Ok
Ok = False
valiPredi=False
st.set_page_config(page_title="Proyecto Trabajo de Titulo", layout="wide", initial_sidebar_state="expanded")
st.title("Proyecto de Modelo Predictivo de la Calidad")

#*  Mostrando los Modelos en pantalla
with st.container(border=True):
    st.subheader("Elija un modelo a trabajar")

    with open(r'data\modelos.json', 'r') as file:
        nombres_modelos = json.load(file)
    st.write("Modelos disponibles:")
    for i, nombre in enumerate(nombres_modelos):
        st.write(f"{i+1}.{nombre}")

#*  Eleccion el Modelo
opcion = st.selectbox("Elección del Modelo", ["","Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "Support Vector Regressor (SVR)", "K-Nearest Neighbors (KNN)"])

with open(r"data\opcion_seleccionada.json", "w") as file:
    json.dump({"opcion": opcion}, file)


#? Esta linea de codigo entrega las columnas de ejemplo con las que se esta trabajando al momento
with open(r'data\columnas_ejemplo.json', 'r') as file:
    columnasUtilizadas = json.load(file)
#st.write(columnasUtilizadas)

if opcion:

    st.write(f"Se ha elegido el modelo:{opcion}")

    #! Valores para prediccion
    st.header("Ingrese los datos necesarios para realizar la predicción: ")
    
    
    # Función para procesar cada entrada
    def procesar_entrada():
        global Ok
        # Diccionario para almacenar las entradas
        entrada = {}


        st.subheader("Fecha del tratamiento")
        # Fila 1: Fecha
        col1 = st.columns(1)[0]
        with col1:
            #! Entrada de fecha  ----------------------------------------------------------------------------------------------------------------------------
            entrada["fecha_registro"] = st.date_input("Ingrese la fecha del tratamiento (formato DD/MM/AAAA):",date.today())
            # Convertir la fecha a DD/MM/YYYY
            entrada["fecha_registro"] = entrada["fecha_registro"].strftime("%Y/%m/%d")
        # Fila 2: Horarios
        st.subheader("Horarios")
        col2, col3 = st.columns(2)
        with col2:
            #! Entrada de horas  ----------------------------------------------------------------------------------------------------------------------------
            entrada["hora_ingreso_al_sistema_1"] = st.time_input("Seleccione la hora de ingreso al sistema (formato HH:MM) [1]:", key="hora_ingreso_al_sistema_1")
            entrada["hora_ingreso_area_planificacion_3"] = st.time_input("3) Seleccione la hora de inicio de la planificación (formato HH:MM) [3]:", key="hora_ingreso_area_planificacion_3")
            entrada["hora_ingreso_area_tratamiento_5"] = st.time_input("5) Seleccione la hora de inicio del tratamiento (formato HH:MM) [5]:", key="hora_ingreso_area_tratamiento_5") 
        with col3:
            entrada["hora_ingreso_area_simualcion_2"] = st.time_input("2) Seleccione la hora de ingreso al área de simulación (formato HH:MM) [2]:", key="hora_ingreso_area_simualcion_2")
            entrada["hora_inicio_preparativos_4"] = st.time_input("4) Seleccione la hora a la que se iniciaron los preparativos (formato HH:MM) [4]:", key="hora_inicio_preparativos_4")
            entrada["hora_de_salida_paciente_6"] = st.time_input("6) Seleccione la hora a la que se retiró el paciente (formato HH:MM) [6]:", key="hora_de_salida_paciente_6")
        
        claves_horas = [key for key in entrada if key.startswith("hora_")]
        # Comparar las horas en secuencia
        def verificar_secuencia_horaria(entrada, claves_horas):
            errores = []
            for i in range(len(claves_horas) - 1):
                hora_actual = entrada[claves_horas[i]]
                hora_siguiente = entrada[claves_horas[i + 1]]

            if hora_siguiente <= hora_actual:
                errores.append("Horas de Ingreso Incorrectas")
            return errores
        # Validar las horas
        errores = verificar_secuencia_horaria(entrada, claves_horas)
        # Mostrar resultados
        if errores:
            for error in errores:
                st.warning(error)
        else:
            st.success("Horas validas")
            Ok=True
        # Convertir las horas a formato HH:MM
        entrada["hora_ingreso_al_sistema_1"] = entrada["hora_ingreso_al_sistema_1"].strftime("%H:%M")
        entrada["hora_ingreso_area_simualcion_2"] = entrada["hora_ingreso_area_simualcion_2"].strftime("%H:%M")
        entrada["hora_ingreso_area_planificacion_3"] = entrada["hora_ingreso_area_planificacion_3"].strftime("%H:%M")
        entrada["hora_inicio_preparativos_4"] = entrada["hora_inicio_preparativos_4"].strftime("%H:%M")
        entrada["hora_ingreso_area_tratamiento_5"] = entrada["hora_ingreso_area_tratamiento_5"].strftime("%H:%M")
        entrada["hora_de_salida_paciente_6"] = entrada["hora_de_salida_paciente_6"].strftime("%H:%M")

        # Fila 3: Ingreso
        st.subheader("Ingreso del paciente")
        col4, col5, col6 = st.columns(3)
        with col4:
            entrada["verificacion_ingreso_paciente_al_sistema"] = 1 if st.toggle("¿Se ingresó al paciente en el sistema?", key="verificacion_ingreso_paciente_al_sistema") else 0
            entrada["confirmacion_informe_ingreso"] = 1 if st.toggle("¿Se confirmó la recepción del informe de ingreso?", key="confirmacion_informe_ingreso") else 0
            entrada["verificacion_actividad_fuente"] = 1 if st.toggle("¿Se verificó la actividad de la fuente?", key="verificacion_actividad_fuente") else 0
        with col5:
            entrada["verificacion_informacion_clinica"] = 1 if st.toggle("¿Se verificó la información clínica necesaria para el tratamiento?", key="verificacion_informacion_clinica") else 0
            entrada["verificacion_identidad_paciente"] = 1 if st.toggle("¿Se verificó la identidad del paciente?", key="verificacion_identidad_paciente") else 0
            entrada["comprobacion_delimitacion_zona"] = 1 if st.toggle("¿Finalizó la delimitación de la zona de aplicación?", key="comprobacion_delimitacion_zona") else 0
        with col6:
            entrada["recepcion_ficha_medica"] = 1 if st.toggle("¿Se recepcionó la ficha médica?", key="recepcion_ficha_medica") else 0
            entrada["comprobacion_ubicacion_aplicadores"] = 1 if st.toggle("¿Se comprobó la correcta ubicación de los aplicadores con el Scanner?", key="comprobacion_ubicacion_aplicadores") else 0
            entrada["comprobacion_indicacion_postura"] = 1 if st.toggle("¿Se le dieron al paciente las indicaciones de postura?", key="comprobacion_indicacion_postura") else 0
        
        # Fila 3: Ingreso
        st.subheader("Planificación")
        col7, col8, col9 = st.columns(3)
        with col7:
            entrada["validacion_informe_ingreso"] = 1 if st.toggle("¿Se comprobó la información del informe de ingreso?", key="validacion_informe_ingreso") else 0
            entrada["finalizacion_analisis_curvas_dosis"] = 1 if st.toggle("¿Se finalizó en análisis de las curvas de dosis?", key="finalizacion_analisis_curvas_dosis") else 0
            entrada["determinacion_lugar_colocacion"] = 1 if st.toggle("¿Se determinó el lugar de colocación para el aplicador?", key="determinacion_lugar_colocacion") else 0
        with col8:
            entrada["comprobacion_repeticion_tratamiento"] = 1 if st.toggle("¿El paciente viene a tratamiento por repetición?", key="comprobacion_repeticion_tratamiento") else 0
            entrada["verificacion_calculo_dosimetrico"] = 1 if st.toggle("¿Se verificó la dósis por medio del cálculo manual?", key="verificacion_calculo_dosimetrico") else 0
        with col9:
            entrada["verificacion_prescripcion_dosis"] = 1 if st.toggle("¿Se verificó la dosis prescrita por el médico?", key="verificacion_prescripcion_dosis") else 0
            entrada["finalizacion_segmentacion_oars"] = 1 if st.toggle("¿Se segmentaron los organos en riesgo?", key="finalizacion_segmentacion_oars") else 0
        
        # Fila 4: Preparativos
        st.subheader("Preparativos")
        col10, col11, col12 = st.columns(3)
        with col10:
            entrada["entrega_consentimiento_informado"] = 1 if st.toggle("¿Se hizo entrega del consentimiento informado?", key="entrega_consentimiento_informado") else 0
            entrada["cantidad_personal_asignado"] = st.number_input("¿Cuanto personal se asignó para este paciente?",step=1,min_value=0,key="cantidad_personal_asignado")
            entrada["validacion_estado_fuente"] = 1 if st.toggle("¿Se revisó el estado de la fuente?", key="validacion_estado_fuente") else 0
            entrada["entrega_medicamentos_necesarios"] = 1 if st.toggle("¿Se le entregaron los medicamentos al paciente en el caso de que fueran requeridos?", key="entrega_medicamentos_necesarios") else 0
            entrada["verificacion_posicion_tubos_transferencia"] = 1 if st.toggle("¿Se comprobó la correcta colocación de los tubos de transferencia?", key="verificacion_posicion_tubos_transferencia") else 0
        with col11:
            entrada["verificacion_diaria_maquinaria"] = 1 if st.toggle("¿Se realizó la verificación de maquinaria diaria?", key="verificacion_diaria_maquinaria") else 0
            entrada["validacion_personal_asignado"] = 1 if st.toggle("¿Todo el personal asignado se encontraba capacitado al momento del tratamiento?", key="validacion_personal_asignado") else 0
            entrada["validacion_correcta_colocacion_aplicador"] = 1 if st.toggle("¿Se verificó la correcta colocación del aplicador?", key="validacion_correcta_colocacion_aplicador") else 0
            entrada["evaluacion_paciente"] = 1 if st.toggle("¿Se realizó la evaluación física del paciente?", key="evaluacion_paciente") else 0
            entrada["verificar_informacion_plan"] = 1 if st.toggle("¿Se comprobó que la información del plan esté completa?", key="verificar_informacion_plan") else 0
        with col12:
            entrada["verificacion_consentimiento_informado"] = 1 if st.toggle("¿Se verificó la correcta recepción del consentimiento informado?", key="verificacion_consentimiento_informado") else 0
            entrada["validacion_tiempo_radiacion"] = 1 if st.toggle("¿Se comprobó cual fue el tiempo final de exposición?", key="validacion_tiempo_radiacion") else 0
            entrada["finalizacion_revision_tolrancia_fuente"] = 1 if st.toggle("¿Se realizó una revisión de la tolerancia de la fuente?", key="finalizacion_revision_tolrancia_fuente") else 0
            entrada["verificacion_controles_de_seguridad"] = 1 if st.toggle("¿Se llevaron a cabo los controles de seguridad correspondientes?", key="verificacion_controles_de_seguridad") else 0
            entrada["verificar_concordancia_tiempo_respecto_a_planificacion"] = 1 if st.toggle("¿Se comparó el tiempo de exposición en la máquina respecto a la planificación?", key="verificar_concordancia_tiempo_respecto_a_planificacion") else 0
        
        # Fila 4: Tratamiento
        st.subheader("Tratamiento")
        col13, col14, col15 = st.columns(3)
        with col13:
            entrada["verificacion_paciente"] = 1 if st.toggle("¿Se comprobó la identidad del paciente antes del tratamiento?", key="verificacion_paciente") else 0
            entrada["registro_fotografico_aplicadores"] = 1 if st.toggle("¿Se guardó registro fotográfico de los aplicadores?", key="registro_fotografico_aplicadores") else 0
        with col14:
            entrada["comprobacion_concordancia_tiemmpos_exposicion"] = 1 if st.toggle("¿Se comparó el tiempo de exposición de la máquina con una toma de tiempo independiente?", key="comprobacion_concordancia_tiemmpos_exposicion") else 0
            entrada["finalizar_checklist_comprobacion"] = 1 if st.toggle("¿Se finalizó el proceso del tratamiento correctamente?", key="finalizar_checklist_comprobacion") else 0
        with col15:
            entrada["registro_aplicadores_utilizados"] = 1 if st.toggle("¿Se registraron los aplicadores utilizados?", key="registro_aplicadores_utilizados") else 0
            entrada["cantidad_reportes_incidentes"] = st.number_input("Ingrese la cantidad de incidentes reportados",step=1,min_value=0,key="cantidad_reportes_incidentes")

        return entrada  # Devolver el diccionario con los valores ingresados

    # Llamar a la función y obtener los valores ingresados
    entrada = procesar_entrada()

#!===================================================================
# Botón para guardar los datos y ejecutar el backend
    if Ok == True:
        if st.button("Enviar"):
                valiPredi=True
                # Guardar los datos en un archivo JSON
                with open(r"data\datos.json", "w") as file:
                    json.dump(entrada, file)

                # Iniciar el backend
                backend_process = subprocess.run(["python", "src\modelo.py"])
                # Cargar la predicción desde el archivo
                # Leer resultados
                try:
                    with open(r'data\predict&proba.json', 'r') as file:
                        resultados = json.load(file)

                except FileNotFoundError:
                    st.error("No se encontraron resultados. Por favor, realice una predicción primero.")

    if valiPredi is True:
    # Mostrar la predicción en un expander
        with st.expander("""*Predicción*"""):
            if prediccion is not None:
                # Mostrar resultados
                st.subheader("Resultados de la Predicción")
                st.write(f"Modelo Utilizado: {opcion}")
                st.write(f"**Predicción del caso:** {resultados['prediccion']} reclamos")
            else:
                st.write("La predicción no está disponible aún.")