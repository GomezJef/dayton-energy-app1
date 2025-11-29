import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuración de la página
st.set_page_config(page_title="Predicción Energía Dayton", layout="wide")
st.title("⚡ Predicción de Consumo Eléctrico - Dayton, Ohio")

# 2. Funciones de Carga (Con caché para velocidad)
@st.cache_data
def cargar_datos():
    # Cargar CSV de consumo
    df = pd.read_csv('DAYTON_hourly.csv', index_col=0, parse_dates=True)
    
    # Renombrar columna si es necesario (Corrección clave)
    if 'DAYTON_MW' in df.columns:
        df = df.rename(columns={'DAYTON_MW': 'Consumo_MW'})
    
    # Cargar CSV de clima
    clima = pd.read_csv('4177229.csv')
    
    # ¡IMPORTANTE! Esta línea devuelve los datos al programa principal
    return df, clima

@st.cache_resource
def cargar_modelo():
    # Intenta cargar con el nombre que usaste en el notebook
    try:
        return joblib.load('modelo_demanda_final.joblib')
    except:
        # Si falla, intenta con el otro nombre posible
        return joblib.load('modelo_consumo_dayton.joblib')

# 3. Bloque Principal de Carga de Datos
try:
    # Llamamos a las funciones y guardamos los datos en variables
    df_main, df_clima = cargar_datos()
    model = cargar_modelo()
    st.success("¡Sistema listo! Datos cargados.")
except Exception as e:
    # Si algo falla, mostramos el error detallado
    st.error(f"Error crítico cargando archivos: {e}")
    st.info("Por favor verifica que los archivos .csv y .joblib estén subidos en GitHub.")
    st.stop()

# 4. Interfaz de Usuario (Sidebar)
st.sidebar.header("Parámetros de Predicción")

fecha_input = st.sidebar.date_input("Seleccionar Fecha")
hora_input = st.sidebar.slider("Seleccionar Hora (0-23)", 0, 23, 12)
temp_input = st.sidebar.number_input("Temperatura Promedio (°C)", value=15.0)
consumo_previo = st.sidebar.number_input("Consumo Hora Anterior (MW)", value=2000.0)

# 5. Botón y Lógica de Predicción
if st.sidebar.button("Calcular Predicción"):
    # Reconstruir fecha para extraer características
    fecha_completa = pd.to_datetime(f"{fecha_input} {hora_input}:00:00")
    
    # Crear dataframe de entrada con las mismas columnas del entrenamiento
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

    try:
        # Si el modelo es un pipeline, él se encarga de escalar si es necesario
        prediccion = model.predict(input_data)[0]

        # Mostrar resultados
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="⚡ Demanda Estimada", value=f"{prediccion:.2f} MW")
        with col2:
            st.info(f"Detalles:\n- Fecha: {fecha_completa}\n- Temp: {temp_input}°C")
            
    except Exception as e:
        st.error(f"Error al predecir: {e}")

# 6. Gráfico Histórico (Verificación final)
st.subheader("Tendencia Reciente de Consumo Real")
try:
    # Graficamos los últimos 500 datos para no sobrecargar
    st.line_chart(df_main['Consumo_MW'].tail(500))
except Exception as e:
    st.warning(f"No se pudo generar el gráfico: {e}")
