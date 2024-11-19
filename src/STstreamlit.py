import streamlit as st
import json
import subprocess
import psutil
from datetime import datetime
import time
import os

# Ruta del archivo de bloqueo
lock_file = 'data\\modelo_lock.lock'

# Verificar si el archivo de bloqueo ya existe
if not os.path.exists(lock_file):
    # Crear el archivo de bloqueo
    with open(lock_file, 'w') as f:
        f.write('Proceso en ejecución')

    # Iniciar el proceso de modelo.py como subproceso
    modelo_proceso = subprocess.Popen(["python", "src\modelo.py"])
else:
    print("modelo.py ya está en ejecución.")

# Lógica de Streamlit aquí

# Al finalizar, eliminar el archivo de bloqueo
# Esto se asegura de que se borre el archivo al salir de Streamlit
if os.path.exists(lock_file):
    os.remove(lock_file)

#*  Mostrar el desempeño de los modelos
st.write("""*Mostrar el desempeño de los modelos*""")
with open(r'data\resultados_df.json', 'r') as file:
    mod = json.load(file)
    
#*  Mostrando los Modelos en pantalla
with st.container():
    with open(r'data\\modelos.json', 'r') as file:
        nombres_modelos = json.load(file)
    st.write("Modelos disponibles:")
    for i, nombre in enumerate(nombres_modelos):
        st.write(f"{i+1}.{nombre}")

#*  Eleccion el Modelo
opcion = st.selectbox("Elección del Modelo", [0, 1, 2, 3, 4, 5], index=0)
with open("data\\opcion_seleccionada.json", "w") as file:
    json.dump({"opcion": opcion}, file)



#? Esta linea de codigo entrega las columnas de ejemplo con las que se esta trabajando al momento
with open(r'data\\columnas_ejemplo.json', 'r') as file:
    columnasUtilizadas = json.load(file)
#st.write(columnasUtilizadas)


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
                # Entrada de hora utilizando time_input
                valor = st.time_input(f"Ingrese el valor para {columna}:", datetime.strptime("00:00", '%H:%M').time())
                # Almacenamos la hora en el formato deseado
                entrada[columna] = str(valor.strftime('%H:%M'))
            elif "Fecha" in columna:
                # Entrada de fecha
                valor_fecha = st.date_input(f"Ingrese el valor para {columna}")
                try:
                    # Intentamos convertir el valor a un string en el formato deseado
                    entrada[columna] = valor_fecha.strftime('%d/%m/%Y')
                except ValueError:
                    # En caso de que haya un error en el formato, se usa un valor por defecto
                    st.warning(f"Formato inválido para {columna}. Se usará el valor por defecto: 01/01/2000.")
                    entrada[columna] = "01/01/2000"
            else:  # Manejar otros valores como números
                valor = st.text_input(f"Ingrese el valor para {columna}")
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
    with open("data\\datos.json", "w") as file:
        json.dump(entrada, file)
    st.success("Datos enviados al backend")
