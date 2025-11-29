import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuración de la página
st.set_page_config(page_title="Predicción Energía Dayton", layout="wide")

st.title("⚡ Predicción de Consumo Eléctrico - Dayton, Ohio")
st.markdown("Esta aplicación utiliza Inteligencia Artificial para estimar la demanda eléctrica basada en el clima y la fecha.")

# 2. Carga de recursos (con caché para que sea rápido)
@st.cache_data
def cargar_datos():
    # Cargar tus CSVs (Asegúrate de que estén en la misma carpeta)
    df = pd.read_csv('DAYTON_hourly.csv', index_col=0, parse_dates=True)
    clima = pd.read_csv('4177229.csv')
    return df, clima

@st.cache_resource
def cargar_modelo():
    return joblib.load('modelo_consumo_dayton.joblib')

try:
    df_main, df_clima = cargar_datos()
    model = cargar_modelo()
    st.success("Datos y Modelo cargados correctamente!")
except Exception as e:
    st.error(f"Error cargando archivos: {e}")
    st.stop()

# 3. Sidebar para Inputs del Usuario
st.sidebar.header("Parámetros de Predicción")

# El usuario elige una fecha y hora futura o hipotética
fecha_input = st.sidebar.date_input("Seleccionar Fecha")
hora_input = st.sidebar.slider("Seleccionar Hora (0-23)", 0, 23, 12)
temp_input = st.sidebar.number_input("Temperatura Promedio (°C)", value=15.0, step=0.5)

# Input importante: El consumo de la hora anterior (Lag)
# En una app real esto se automatizaría, pero aquí lo pedimos manual o ponemos un default
consumo_previo = st.sidebar.number_input("Consumo Hora Anterior (MW)", value=2000.0, step=10.0)

# 4. Lógica de Predicción
if st.sidebar.button("Calcular Predicción"):
    # Crear un pequeño dataframe con los datos de entrada
    # IMPORTANTE: Debe tener las mismas columnas que usaste para entrenar (X)
    
    # Reconstruimos las features de fecha
    fecha_completa = pd.to_datetime(f"{fecha_input} {hora_input}:00:00")
    
    input_data = pd.DataFrame({
        'hora': [hora_input],
        'dia_semana': [fecha_completa.dayofweek],
        'mes': [fecha_completa.month],
        'año': [fecha_completa.year],
        'semana_del_año': [fecha_completa.isocalendar().week],
        'es_fin_de_semana': [1 if fecha_completa.dayofweek >= 5 else 0],
        'Consumo_Lag_1': [consumo_previo],
        'TAVG': [temp_input]
    })

    # Predecir
    prediccion = model.predict(input_data)[0]

    # 5. Mostrar Resultados
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="⚡ Demanda Predicha", value=f"{prediccion:.2f} MW")
    
    with col2:
        st.info(f"Fecha: {fecha_completa}\n\nTemp: {temp_input}°C")

# 6. Visualización de datos históricos (Contexto)
st.subheader("Tendencia Histórica de Consumo")
st.line_chart(df_main['Consumo_MW'].tail(500))  # Muestra las últimas 500 horas reales
