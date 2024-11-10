import streamlit as st
import pandas as pd
import json

#*  Mostrar el desempeño de los modelos
st.write("""*Mostrar el desempeño de los modelos*""")
mod = pd.read_json(r'C:\\Users\\Gustavo\\Desktop\\TrabajoDeTitulo\\prototipoModeloDeCalidad\\resultados_df.json')
st.write(mod)

#*  Mostrando los Modelos en pantalla
with st.container():
    with open(r'C:\Users\Gustavo\Desktop\TrabajoDeTitulo\prototipoModeloDeCalidad\modelos.json', 'r') as file:
        nombres_modelos = json.load(file)
    st.write("Modelos disponibles:")
    st.write(nombres_modelos)

#*  Eleccion el Modelo
opcion = st.selectbox("Elección del Modelo", [0, 1, 2, 3, 4, 5], index=0)
with open("opcion_seleccionada.json", "w") as file:
    json.dump({"opcion": opcion}, file)

#! Valores para prediccion
st.write("Ingrese los datos necesarios para realizar la predicción: ")
with open(r'C:\Users\Gustavo\Desktop\TrabajoDeTitulo\prototipoModeloDeCalidad\columnas_ejemplo.json', 'r') as file:
    columnasUtilizadas = json.load(file)



#!Creacion de los botones
st.write("""Parte de variables""")
columna=st.columns(1)[0]
with columna:
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
