import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
import xgboost as xgb

# Función para dibujar la cancha de fútbol
def draw_pitch(ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))

    # Pitch Outline & Centre Line
    plt.plot([0, 0, 100, 100, 0], [0, 100, 100, 0, 0], color="black")

    # Left Penalty Area
    plt.plot([0, 17, 17, 0], [30, 30, 70, 70], color="black")

    # Right Penalty Area
    plt.plot([100, 83, 83, 100], [30, 30, 70, 70], color="black")

    # Left 6-yard Box
    plt.plot([0, 6, 6, 0], [44, 44, 56, 56], color="black")

    # Right 6-yard Box
    plt.plot([100, 94, 94, 100], [44, 44, 56, 56], color="black")

    # Prepare Circles; 10 yard circle at centre
    centreCircle = plt.Circle((50, 50), 8, color="black", fill=False)
    centreSpot = plt.Circle((50, 50), 0.8, color="black")
    leftPenSpot = plt.Circle((11, 50), 0.8, color="black")
    rightPenSpot = plt.Circle((89, 50), 0.8, color="black")

    # Draw Circles
    ax.add_patch(centreCircle)
    ax.add_patch(centreSpot)
    ax.add_patch(leftPenSpot)
    ax.add_patch(rightPenSpot)

    # Prepare Arcs
    leftArc = Arc((11, 50), height=16.2, width=16.2, angle=0, theta1=308, theta2=52, color="black")
    rightArc = Arc((89, 50), height=16.2, width=16.2, angle=0, theta1=128, theta2=232, color="black")

    # Draw Arcs
    ax.add_patch(leftArc)
    ax.add_patch(rightArc)
    
    # Tidy Axes
    plt.axis('off')

    return ax

# Cargar los datos de CSV
file_path = '/mnt/data/Base_Anonimizada.csv'

try:
    df = pd.read_csv(file_path, delimiter=';', encoding='utf-8')
    st.write("Archivo leído correctamente como CSV.")
except Exception as e:
    st.write(f"Error al leer el archivo: {e}")
    df = pd.DataFrame()  # Crear un dataframe vacío en caso de error

# Cargar los modelos XGBoost
modelos = {}
targets = [
    'avg_dist_sess_m', 'zona_4_19.9_25.1_kmh',
    'zona_5_mas_25.1_kmh', 'num_aceleraciones_intensas',
    'num_desaceleraciones_intensas', 'num_acel_desintensas',
    'num_sprints_total', 'prom_esfuerzos_repetidos', 'max_vel_kmh'
]

for target in targets:
    modelo_path = f'modelo_xgboost_{target}.json'
    try:
        modelo = xgb.XGBRegressor()
        modelo.load_model(modelo_path)
        modelos[target] = modelo
        st.write(f"Modelo XGBoost para {target} cargado correctamente.")
    except Exception as e:
        st.write(f"Error al cargar el modelo XGBoost para {target}: {e}")
        modelos[target] = None

# Comprobar que el dataframe no está vacío y los modelos están cargados
if not df.empty and all(modelos.values()):
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

    # Seleccionar la categoría del partido de la columna
    categorias = df['Categoria de Partido'].unique()
    categoria = st.selectbox('Selecciona la categoría del partido:', categorias)

    # Input para el umbral
    umbral = st.number_input('Umbral', min_value=0.0, max_value=1.0, value=0.5)

    # Función para preparar los datos y hacer la predicción
    def preparar_datos_y_predecir(jugador_info, categoria, umbral, modelos):
        # Crear un DataFrame con la nueva fila
        nueva_fila = pd.DataFrame({
            'categoria_partido': [categoria],
            'Altura': [jugador_info['Altura']],
            'Peso': [jugador_info['Peso']],
            'Gol?': [jugador_info['Gol?']],
            'Asistencia?': [jugador_info['Asistencia?']],
            'Minutos': [jugador_info['Minutos']],
            'Umbral': [umbral]
        })
        
        # Asegurar que las columnas estén en el mismo orden que las usadas para entrenar el modelo
        nueva_fila = pd.get_dummies(nueva_fila, columns=['categoria_partido'], drop_first=True)
        resultados = {}
        
        for target in modelos:
            modelo = modelos[target]
            if modelo:
                # Asegurar que todas las características necesarias están presentes
                dmatrix = xgb.DMatrix(nueva_fila)
                prediccion = modelo.predict(dmatrix)
                resultados[target] = prediccion[0]
        
        return resultados

    # Inicializar lista de jugadores en el campo
    if 'jugadores_campo' not in st.session_state:
        st.session_state.jugadores_campo = []

    # Botón para predecir y agregar jugador
    if st.button('Predecir y Agregar al Campo'):
        resultados = preparar_datos_y_predecir(jugador_info, categoria, umbral, modelos)
        st.write(f"Resultados de predicción: {resultados}")
        if resultados['minutos'] >= umbral:
            st.session_state.jugadores_campo.append((jugador_info['Posición Habitual'], jugador_info['Jugador anonimizado']))
            st.write("Jugador agregado al campo.")

    # Mostrar la lista de jugadores en el campo
    st.write("**Jugadores en el campo:**")
    for pos, j in st.session_state.jugadores_campo:
        st.write(f"{pos}: Jugador {j}")

    # Visualizar un campo de juego con los jugadores agregados
    st.write("**Visualización del campo de juego:**")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax = draw_pitch(ax)
    
    # Posiciones aproximadas para visualización
    posiciones = {
        'Portero': (10, 50),
        'Defensa': (30, 50),
        'Lateral / Volante': (30, 30),
        'Volante / Extremo': (50, 50),
        'Centrocampista': (50, 30),
        'Delantero': (70, 50)
    }

    for pos, j in st.session_state.jugadores_campo:
        if pos in posiciones:
            x, y = posiciones[pos]
            ax.text(x, y, f'{j}', fontsize=12, ha='center', va='center', bbox=dict(facecolor='red', alpha=0.5))

    st.pyplot(fig)
else:
    st.write("No se pudieron cargar los datos o los modelos.")
