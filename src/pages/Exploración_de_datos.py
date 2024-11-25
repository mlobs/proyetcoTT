import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Cargar el dataset
data = pd.read_csv(r'data\DatasetTT.csv')

st.write("Exploración de Datos")

correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()