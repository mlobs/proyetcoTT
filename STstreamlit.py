import streamlit as st
import pandas as pd
import json
import subprocess
from datetime import datetime
#!           streamlit run c:/Users/Gustavo/Desktop/TrabajoDeTitulo/prototipoModeloDeCalidad/STstreamlit.py


backend_process = subprocess.Popen(["python", "modelo.py"])
st.write("Frontend con Streamlit está corriendo...")

#*  Mostrar el desempeño de los modelos
st.write("""*Mostrar el desempeño de los modelos*""")
mod = pd.read_json(r'resultados_df.json')
st.write(mod)

#*  Mostrando los Modelos en pantalla
with st.container():
    with open(r'C:\Users\Gustavo\Desktop\TrabajoDeTitulo\prototipoModeloDeCalidad\modelos.json', 'r') as file:
        nombres_modelos = json.load(file)
    st.write("Modelos disponibles:")
    for i, nombre in enumerate(nombres_modelos):
        st.write(f"{i+1}.{nombre}")

#*  Eleccion el Modelo
opcion = st.selectbox("Elección del Modelo", [0, 1, 2, 3, 4, 5], index=0)
with open("opcion_seleccionada.json", "w") as file:
    json.dump({"opcion": opcion}, file)



#? Esta linea de codigo entrega las columnas de ejemplo con las que se esta trabajando al momento
with open(r'columnas_ejemplo.json', 'r') as file:
    columnasUtilizadas = json.load(file)
st.write(columnasUtilizadas)


if opcion:
    for i, nombre in enumerate(nombres_modelos):
        if opcion-1 == i: 
            st.write(f"Se ha elegido el modelo:{nombre}")

    #! Valores para prediccion
    st.write("Ingrese los datos necesarios para realizar la predicción: ")
    
    
# Función para procesar cada entrada
    def procesar_entrada(columnas):
        # Diccionario para almacenar las entradas
        entrada = {}
        
        # Recorrer cada columna y solicitar la entrada de datos
        for columna in columnas:
            if "Hora" in columna:
                # Entrada de hora
                valor = st.text_input(f"Ingrese el valor para {columna} (formato HH:MM):", "00:00")
                try:
                    entrada[columna] = datetime.strptime(valor, '%H:%M').strftime('%H:%M')
                except ValueError:
                    st.warning(f"Formato inválido para {columna}. Se usará el valor por defecto: 00:00.")
                    entrada[columna] = "00:00"
            elif "Fecha" in columna:
                # Entrada de fecha
                valor = st.text_input(f"Ingrese el valor para {columna} (formato DD/MM/AAAA):", "01/01/2000")
                try:
                    entrada[columna] = datetime.strptime(valor, '%d/%m/%Y').strftime('%d/%m/%Y')
                except ValueError:
                    st.warning(f"Formato inválido para {columna}. Se usará el valor por defecto: 01/01/2000.")
                    entrada[columna] = "01/01/2000"
            else:  # Manejar otros valores como números
                valor =st.text_input(f"Ingrese el valor para {columna}")
                try:
                    entrada[columna] = float(valor.strip()) if valor is not None and valor.strip() != "" else 0.0
                except ValueError:
                    st.write(f"Valor inválido para {columna}. Se usará el valor por defecto: 0.0")
                    entrada[columna] = 0.0

            
            
        entrada[f"{columna}_Infoclinica"] = st.checkbox("Se verificó la información clínica necesaria?", key=f"{columna}_Infoclinica")
        entrada[f"{columna}_TratamientoRepeticion"] = st.checkbox("Comprobación Tratamiento por Repetición", key=f"{columna}_TratamientoRepeticion")
        entrada[f"{columna}_PrescripcionDosis"] = st.checkbox("Verificación de Prescripción de dosis", key=f"{columna}_PrescripcionDosis")
        entrada[f"{columna}_FinalizaAnalisisCurvasDosis"] = st.checkbox("Finalización Análisis de Curvas de dosis", key=f"{columna}_FinalizaAnalisisCurvasDosis")
        entrada[f"{columna}_VerifCalculoDosimetrico"] = st.checkbox("Verificación de cálculo dosimétrico", key=f"{columna}_VerifCalculoDosimetrico")
        entrada[f"{columna}_FinalizaSegmentacionOARs"] = st.checkbox("Finalización Segmentación de OARs", key=f"{columna}_FinalizaSegmentacionOARs")
        entrada[f"{columna}_LugarColocacion"] = st.checkbox("Determinación de lugar de colocación", key=f"{columna}_LugarColocacion")
        entrada[f"{columna}_EntregaConsentimiento"] = st.checkbox("Entrega consentimiento Informado", key=f"{columna}_EntregaConsentimiento")
        entrada[f"{columna}_VerifDiariaMaquina"] = st.checkbox("Verificación diaria de Maquina", key=f"{columna}_VerifDiariaMaquina")
        entrada[f"{columna}_VerifConsentimientoInformado"] = st.checkbox("Verificación Consentimiento Informado", key=f"{columna}_VerifConsentimientoInformado")
        entrada[f"{columna}_ValidaPersonalAsignado"] = st.checkbox("Validación de Personal asignado", key=f"{columna}_ValidaPersonalAsignado")
        entrada[f"{columna}_CantidadPersonalAsignado"] = st.checkbox("Cantidad de Personal asignado", key=f"{columna}_CantidadPersonalAsignado")
        entrada[f"{columna}_ValidaTiempoRadiacion"] = st.checkbox("Validación Tiempo de Radiación", key=f"{columna}_ValidaTiempoRadiacion")
        entrada[f"{columna}_ValidaEstadoFuente"] = st.checkbox("Validación Estado de la Fuente", key=f"{columna}_ValidaEstadoFuente")
        entrada[f"{columna}_ColocacionAplicador"] = st.checkbox("Validar Correcta colocación del aplicador", key=f"{columna}_ColocacionAplicador")
        entrada[f"{columna}_FinalizaRevisionToleranciaFuente"] = st.checkbox("Finalización Revisión de tolerancia de la fuente", key=f"{columna}_FinalizaRevisionToleranciaFuente")
        entrada[f"{columna}_EntregaMedicamentos"] = st.checkbox("Entrega de medicamentos necesarios", key=f"{columna}_EntregaMedicamentos")
        entrada[f"{columna}_EvaluacionPaciente"] = st.checkbox("Evaluación del paciente", key=f"{columna}_EvaluacionPaciente")
        entrada[f"{columna}_VerifControlesCalidad"] = st.checkbox("Verificación de los controles de calidad", key=f"{columna}_VerifControlesCalidad")
        entrada[f"{columna}_VerifPosicionTubosTransferencia"] = st.checkbox("Verificación posición tubos de transferencia", key=f"{columna}_VerifPosicionTubosTransferencia")            
        entrada[f"{columna}_VerifInfoPlanCorrecta"] = st.checkbox("Verificar que la información del plan sea la correcta", key=f"{columna}_VerifInfoPlanCorrecta")
        entrada[f"{columna}_VerifPaciente"] = st.checkbox("Verificación de paciente", key=f"{columna}_VerifPaciente")
        entrada[f"{columna}_ComprobacionTiempoRadiacion"] = st.checkbox("Comprobación tiempo de Radiación vs Tiempo de Radiación en máquina", key=f"{columna}_ComprobacionTiempoRadiacion")
        entrada[f"{columna}_RegistroAplicadores"] = st.checkbox("Registro Aplicadores Utilizado", key=f"{columna}_RegistroAplicadores")            
        entrada[f"{columna}_RegistroFotograficoAplicador"] = st.checkbox("Registro fotográfico del Aplicador", key=f"{columna}_RegistroFotograficoAplicador")
        entrada[f"{columna}_FinalizaChecklist"] = st.checkbox("Finalizar Checklist y realizar comprobación", key=f"{columna}_FinalizaChecklist")



        return entrada  # Devolver el diccionario con los valores ingresados

    # Llamar a la función y obtener los valores ingresados
    entrada = procesar_entrada(columnasUtilizadas)

    # Botón para guardar los datos en un archivo JSON
    if st.button("Enviar"):
        with open("valor.json", "w") as file:
            json.dump(entrada, file)
        st.success("Datos enviados al backend")
