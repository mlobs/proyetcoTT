import streamlit as st
import pandas as pd

st.write("PAGINA 1")

st.write("""*Desempeño de los diferentes Modelos*""")
mod = pd.read_json(r'data\resultados_df.json')
st.write(mod)
