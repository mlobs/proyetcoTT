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
mod = pd.read_json(r'C:\\Users\\Gustavo\\Desktop\\TrabajoDeTitulo\\prototipoModeloDeCalidad\\resultados_df.json')
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
with open(r'C:\Users\Gustavo\Desktop\TrabajoDeTitulo\prototipoModeloDeCalidad\columnas_ejemplo.json', 'r') as file:
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
            else:
                Infoclinica = st.checkbox("Se verificó la información clínica necesaria?")              #permanece en estado FALSE mientras no se checkee
                if Infoclinica:
                    st.write("OK")
                ingresoPaciente=st.checkbox('Paciente ingresado a sistema?')

                fichaMed=st.checkbox('Ficha medica recepcionada')

                InfIngr=st.checkbox('Confirmación Informe de Ingreso')

                idPaciente=st.checkbox('Verificación identidad de paciente')

                VerifActividadFuente = st.checkbox('Verificación de actividad de fuente')

                UbicacionAplicadores = st.checkbox('Comprobación de ubicación de aplicadores')

                DelimitacionZona = st.checkbox('Comprobación de delimitación de zona')

                IndicacionPostura = st.checkbox('Comprobación de indicación de postura')

                ValidaInformeIngreso = st.checkbox('Validación Informe de Ingreso')

                TratamientoRepeticion = st.checkbox('Comprobación Tratamiento por Repetición')
                
                PrescripcionDosis = st.checkbox('Verificación de Prescripción de dosis')

                FinalizaAnalisisCurvasDosis = st.checkbox('Finalización Análisis de Curvas de dosis')

                VerifCalculoDosimetrico = st.checkbox('Verificación de cálculo dosimétrico')

                FinalizaSegmentacionOARs = st.checkbox('Finalización Segmentación de OARs')

                LugarColocacion = st.checkbox('Determinación de lugar de colocación')

                EntregaConsentimiento = st.checkbox('Entrega consentimiento Informado')

                VerifDiariaMaquina = st.checkbox('Verificación diaria de Maquina')

                VerifConsentimientoInformado = st.checkbox('Verificación Consentimiento Informado')

                ValidaPersonalAsignado = st.checkbox('Validación de Personal asignado')

                CantidadPersonalAsignado = st.checkbox('Cantidad de Personal asignado')

                ValidaTiempoRadiacion = st.checkbox('Validación Tiempo de Radiación')

                ValidaEstadoFuente = st.checkbox('Validación Estado de la Fuente')

                ColocacionAplicador = st.checkbox('Validar Correcta colocación del aplicador')

                FinalizaRevisionToleranciaFuente = st.checkbox('Finalización Revisión de tolerancia de la fuente')

                EntregaMedicamentos = st.checkbox('Entrega de medicamentos necesarios')

                EvaluacionPaciente = st.checkbox('Evaluación del paciente')

                VerifControlesCalidad = st.checkbox('Verificación de los controles de calidad')

                VerifPosicionTubosTransferencia = st.checkbox('Verificación posición tubos de transferencia')

                VerifInfoPlanCorrecta = st.checkbox('Verificar que la información del plan sea la correcta')

                VerifPaciente = st.checkbox('Verificación de paciente')

                ComprobacionTiempoRadiacion = st.checkbox('Comprobación tiempo de Radiación vs Tiempo de Radiación en máquina')

                RegistroAplicadores = st.checkbox('Registro Aplicadores Utilizado')            
                
                RegistroFotograficoAplicador = st.checkbox('Registro fotográfico del Aplicador')

                FinalizaChecklist = st.checkbox('Finalizar Checklist y realizar comprobación')

        return entrada  # Devolver el diccionario con los valores ingresados

    # Llamar a la función y obtener los valores ingresados
    entrada = procesar_entrada(columnasUtilizadas)

    # Botón para guardar los datos en un archivo JSON
    if st.button("Enviar"):
        with open("valor.json", "w") as file:
            json.dump(entrada, file)
        st.success("Datos enviados al backend")


    #!Creacion de los botones
