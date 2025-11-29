import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Configuración de la página
st.set_page_config(page_title="Predicción Energía Dayton", layout="wide")

st.title("⚡ Predicción Avanzada: Clima y Energía")
st.markdown("""
Esta aplicación utiliza Inteligencia Artificial para:
1. **Estimar la temperatura** del día seleccionado.
2. **Predecir el consumo eléctrico** basado en esa temperatura y la fecha.
""")

# 2. Carga de recursos (Modelos y Datos)
@st.cache_data
def cargar_datos_clima():
    # Cargamos el histórico para buscar datos reales si existen
    try:
        clima = pd.read_csv('4177229.csv')
        clima['DATE'] = pd.to_datetime(clima['DATE'])
        # Rellenar nulos para tener datos completos por si acaso
        clima['TAVG'] = clima['TAVG'].fillna((clima['TMAX'] + clima['TMIN']) / 2)
        return clima
    except FileNotFoundError:
        st.error("No se encontró el archivo '4177229.csv'. Asegúrate de subirlo.")
        return pd.DataFrame()

@st.cache_resource
def cargar_modelos():
    # Cargar Modelo de Energía
    # VERIFICA QUE ESTE NOMBRE COINCIDA CON TU ARCHIVO DE ENERGÍA
    try:
        model_energia = joblib.load('modelo_consumo_dayton.joblib')
    except:
        try:
            model_energia = joblib.load('modelo_demanda_final.joblib')
        except:
            st.error("⚠️ Falta el modelo de energía. Sube 'modelo_consumo_dayton.joblib'.")
            st.stop()
            
    # Cargar Modelo de Clima
    try:
        model_clima = joblib.load('modelo_clima_dayton.joblib')
    except:
        st.warning("⚠️ No se encontró 'modelo_clima_dayton.joblib'. La predicción automática de temperatura no funcionará hasta que lo subas.")
        model_clima = None
        
    return model_energia, model_clima

# Cargamos todo al inicio
df_clima_hist = cargar_datos_clima()
model_energia, model_clima = cargar_modelos()


# --- BARRA LATERAL (INPUTS) ---
st.sidebar.header("Parámetros de Entrada")

# Input 1: Fecha y Hora
fecha_input = st.sidebar.date_input("Seleccionar Fecha")
hora_input = st.sidebar.slider("Seleccionar Hora (0-23)", 0, 23, 12)

# --- LÓGICA INTELIGENTE DE TEMPERATURA ---
temp_sugerida = 15.0 # Valor por defecto
fuente_temp = "Valor por defecto"

if not df_clima_hist.empty:
    # 1. Buscar en histórico
    dato_historico = df_clima_hist[df_clima_hist['DATE'] == pd.to_datetime(fecha_input)]
    
    if not dato_historico.empty:
        temp_sugerida = float(dato_historico.iloc[0]['TAVG'])
        fuente_temp = "Dato Histórico Real"
    elif model_clima is not None:
        # 2. Si no está en histórico, usar IA
        dia_del_anio = pd.to_datetime(fecha_input).dayofyear
        # El modelo espera un DataFrame o array 2D
        prediccion_temp = model_clima.predict([[dia_del_anio]])
        temp_sugerida = float(prediccion_temp[0])
        fuente_temp = "Predicción IA (Modelo Clima)"

# Mostrar info al usuario
st.sidebar.subheader("Condiciones Climáticas")
if fuente_temp != "Valor por defecto":
    st.sidebar.info(f"🌡️ {fuente_temp}: Se sugiere **{temp_sugerida:.1f} °C**")

# Input 2: Temperatura (Permitimos al usuario modificar la sugerencia)
temp_input = st.sidebar.number_input("Temperatura (°C) - Confirmar", value=temp_sugerida, step=0.5)

# Input 3: Lag
st.sidebar.subheader("Estado de la Red")
consumo_previo = st.sidebar.number_input("Consumo Hora Anterior (MW)", value=2000.0, step=10.0)


# --- PREDICCIÓN DE ENERGÍA ---
if st.button("Calcular Predicción de Energía", type="primary"):
    
    # Preparar datos para el modelo de energía
    fecha_completa = pd.to_datetime(f"{fecha_input} {hora_input}:00:00")
    
    # Construir el DataFrame exactamente como se entrenó el modelo
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

    # Predecir Energía
    try:
        prediccion_mw = model_energia.predict(input_data)[0]

        # --- MOSTRAR RESULTADOS ---
        st.divider()
        st.subheader("Resultados del Análisis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(label="📅 Fecha y Hora", value=f"{fecha_input} {hora_input}:00")
        
        with col2:
            st.metric(label="🌡️ Temperatura Usada", value=f"{temp_input:.1f} °C", delta=fuente_temp)
        
        with col3:
            st.metric(label="⚡ Consumo Predicho", value=f"{prediccion_mw:.2f} MW")
            
        # Mensaje final
        st.success(f"La demanda energética estimada es de **{prediccion_mw:.0f} MW**.")
        
    except Exception as e:
        st.error(f"Error al realizar la predicción: {e}")
        st.write("Verifica que las columnas del DataFrame coincidan con las del entrenamiento.")
