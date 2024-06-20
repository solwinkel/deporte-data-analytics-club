import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt


# Page title
st.set_page_config(page_title='ML Model Building', page_icon='🤖')
st.title('🤖 Modelo Predictivo para Cansancio - Analitica del Deporte')

# Cargar los datos de Excel
df = pd.read_excel('/mnt/data/Base_Anonimizada.xlsx')

# Crear un selector de jugador
jugadores = df['Jugador anonimizado'].unique()
jugador_seleccionado = st.selectbox('Selecciona un jugador:', jugadores)

# Mostrar las características del jugador seleccionado
jugador_info = df[df['Jugador anonimizado'] == jugador_seleccionado].iloc[0]

st.write(f"**Altura:** {jugador_info['Altura']} cm")
st.write(f"**Peso:** {jugador_info['Peso']} kg")
st.write(f"**Goles:** {jugador_info['Gol?']}")
st.write(f"**Asistencias:** {jugador_info['Asistencia?']}")
st.write(f"**Partidos Jugados:** {jugador_info['Minutos'] // 90}")
st.write(f"**Posición Habitual:** {jugador_info['Posición Habitual']}")

# Seleccionar la categoría del partido y la posición del jugador
categoria = st.selectbox('Selecciona la categoría del partido:', ['Amistoso', 'Competitivo', 'Importante'])
posicion = st.selectbox('Selecciona la posición en la que jugará:', ['Portero', 'Defensa', 'Centrocampista', 'Delantero'])

# Función para el modelo predictivo (placeholder)
def modelo_predictivo(categoria, posicion, jugador_info):
    # Aquí iría el código de tu modelo predictivo
    return {'probabilidad_éxito': 0.85}

# Ejecutar el modelo predictivo
resultado = modelo_predictivo(categoria, posicion, jugador_info)

# Mostrar los resultados
st.write("**Resultados del Modelo Predictivo:**")
st.write(f"Probabilidad de éxito: {resultado['probabilidad_éxito']*100}%")

# Visualizar un campo de juego (placeholder)
st.write("**Visualización del campo de juego:**")
st.image("https://upload.wikimedia.org/wikipedia/commons/3/39/Football_pitch.svg", width=600)
