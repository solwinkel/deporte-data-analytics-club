import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy import stats

# Cargar datos
train_df = pd.read_excel('train_df.xlsx')
test_df = pd.read_excel('test_df.xlsx')

# Definir targets y otros parámetros
targets = {
    'minutos': (0, 100),
    'avg_dist_sess_m': (0, 12000),
    'zona_4_19.9_25.1_kmh': (0, 1122),
    'zona_5_mas_25.1_kmh': (0, 542),
    'num_aceleraciones_intensas': (0, 120),
    'num_desaceleraciones_intensas': (0, 140),
    'num_acel_desintensas': (0, 250),
    'num_sprints_total': (0, 30),
    'prom_esfuerzos_repetidos': (0, 30),
    'max_vel_kmh': (0, 33)
}

categorical_cols = [
    'torneo',
    'categoria_partido',
    'posicion_habitual'
]

drop_cols = ['fecha', 'num_fecha_torneo', 'posicion', 'rival', 'tiempo']

# Eliminar columnas no necesarias
train_df = train_df.drop(columns=drop_cols, errors='ignore')
test_df = test_df.drop(columns=drop_cols, errors='ignore')

# Realizar codificación one-hot
train_df = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)
test_df = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)

# Asegurarse de que las columnas en test_df coincidan con las de train_df, sin incluir las columnas de targets
missing_cols = set(train_df.columns) - set(test_df.columns)
for col in missing_cols:
    if col not in targets:
        test_df[col] = 0

# Reordenar las columnas de test_df para que coincidan con train_df (menos las columnas de targets)
test_df = test_df[train_df.drop(columns=targets.keys()).columns]

# Entrenar modelos y guardarlos
best_params = {
    'colsample_bytree': 0.8,
    'learning_rate': 0.1,
    'max_depth': 3,
    'min_child_weight': 3,
    'n_estimators': 50,
    'subsample': 0.6
}

# Entrenar modelos y guardarlos
for target in targets:
    lower_limit, upper_limit = targets[target]
    features = [col for col in train_df.columns if col != target and col not in targets]

    X_train = train_df[features]
    y_train = train_df[target]
#     final_model = XGBRegressor(**best_params, random_state=42)
#     final_model.fit(X_train, y_train)
#     final_model.save_model(f'modelo_xgboost_{target}.json')

# Función para calcular el intervalo de confianza del 99%
def confidence_interval(predictions, confidence=0.99):
    mean_pred = np.mean(predictions)
    stderr = stats.sem(predictions)
    margin = stderr * stats.t.ppf((1 + confidence) / 2., len(predictions) - 1)
    return mean_pred - margin, mean_pred + margin

# Nombres legibles para los targets
targets = {
    'minutos': (0, 100),
    'avg_dist_sess_m': (0, 12000),
    'zona_4_19.9_25.1_kmh': (0, 1122),
    'zona_5_mas_25.1_kmh': (0, 542),
    'num_aceleraciones_intensas': (0, 120),
    'num_desaceleraciones_intensas': (0, 140),
    'num_acel_desintensas': (0, 250),
    'num_sprints_total': (0, 30),
    'prom_esfuerzos_repetidos': (0, 30),
    'max_vel_kmh': (0, 33)
}

target_names = {
    'minutos': 'Minutos Jugados',
    'avg_dist_sess_m': 'Distancia Promedio (m)',
    'zona_4_19.9_25.1_kmh': 'Zona 4 (19.9-25.1 km/h)',
    'zona_5_mas_25.1_kmh': 'Zona 5 (>25.1 km/h)',
    'num_aceleraciones_intensas': 'Aceleraciones Intensas',
    'num_desaceleraciones_intensas': 'Desaceleraciones Intensas',
    'num_acel_desintensas': 'Acel/Desintensas',
    'num_sprints_total': 'Sprints Totales',
    'prom_esfuerzos_repetidos': 'Esfuerzos Repetidos',
    'max_vel_kmh': 'Velocidad Máxima (km/h)'
}

# CSS para agregar bordes a las columnas y estilizar la cancha
st.markdown(
    """
    <style>
    .column {
        padding: 10px;
        border: 1px solid #d3d3d3;
        border-radius: 5px;
    }
    .cancha {
        width: 100%;
        background-color: #228B22;
        position: relative;
        border: 2px solid #fff;
        border-radius: 10px;
        margin: auto;
    }
    .jugador {
        width: 50px;
        height: 50px;
        background-color: #FFD700;
        border-radius: 50%;
        text-align: center;
        line-height: 50px;
        position: absolute;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Interfaz de Streamlit
st.title('Predicciones de métricas físicas para jugadores')

# Seleccionar un jugador y tipo de partido
jugadores = test_df['jugador_anonimizado'].unique()
tipos_partido = ['Importante', 'Normal']  # Definimos las categorías directamente

# Crear columnas para el layout
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="column">', unsafe_allow_html=True)
    jugador_seleccionado = st.selectbox('Selecciona un jugador', jugadores)
    tipo_partido_seleccionado = st.selectbox('Selecciona un tipo de partido', tipos_partido)
    st.markdown('</div>', unsafe_allow_html=True)

# Filtrar datos del jugador seleccionado
jugador_data = test_df[test_df['jugador_anonimizado'] == jugador_seleccionado].copy()

# Obtener y mostrar información adicional del jugador
info_jugador = train_df[train_df['jugador_anonimizado'] == jugador_seleccionado].iloc[0]

with col2:
    st.markdown('<div class="column">', unsafe_allow_html=True)
    st.write(f"Edad: {info_jugador['edad']}")
    st.write(f"Altura: {info_jugador['altura']} cm")
    st.write(f"Peso: {info_jugador['peso']} kg")

    # Calcular la cantidad de partidos en los que ha participado
    num_partidos = len(train_df[train_df['jugador_anonimizado'] == jugador_seleccionado]) + \
                   len(test_df[test_df['jugador_anonimizado'] == jugador_seleccionado])
    st.write(f"Cantidad de partidos en los que ha participado: {num_partidos}")
    st.markdown('</div>', unsafe_allow_html=True)

# Almacenar el equipo titular
if 'equipo_titular' not in st.session_state:
    st.session_state.equipo_titular = []

# Funcionalidad para agregar un jugador al equipo titular
if st.button('Agregar al equipo'):
    if len(st.session_state.equipo_titular) < 10:
        posicion = info_jugador['posicion_habitual']
        jugador_info = {
            'jugador': jugador_seleccionado,
            'posicion': posicion
        }
        st.session_state.equipo_titular.append(jugador_info)

# Funcionalidad para eliminar un jugador del equipo titular
if st.button('Eliminar al equipo'):
    if st.session_state.equipo_titular:
        st.session_state.equipo_titular.pop()

# Mostrar la cancha y los jugadores
st.markdown('<div class="cancha">', unsafe_allow_html=True)
for jugador in st.session_state.equipo_titular:
    posicion = jugador['posicion']
    # Ajustar las posiciones en la cancha (deberías ajustar estos valores según el layout deseado)
    posicion_x = 50  # Ejemplo de coordenadas X
    posicion_y = 50  # Ejemplo de coordenadas Y
    st.markdown(
        f'<div class="jugador" style="top: {posicion_y}px; left: {posicion_x}px;">{jugador["jugador"]}</div>',
        unsafe_allow_html=True
    )
st.markdown('</div>', unsafe_allow_html=True)

# Eliminar columnas no necesarias
jugador_data = jugador_data.drop(columns=['jugador_anonimizado'] + drop_cols + list(targets.keys()), errors='ignore')

# Agregar la categoría de partido codificada manualmente
if tipo_partido_seleccionado == 'Normal':
    jugador_data['categoria_partido_Normal'] = 1
else:
    jugador_data['categoria_partido_Normal'] = 0

# Asegurarse de que las columnas de jugador_data coincidan con las del modelo
missing_cols = set(train_df.drop(columns=targets.keys()).columns) - set(jugador_data.columns)
for col in missing_cols:
    jugador_data[col] = 0

# Reordenar las columnas de jugador_data para que coincidan con X_train
jugador_data = jugador_data[train_df.drop(columns=targets.keys()).columns]

# Botón para realizar predicciones
if st.button('Realizar predicciones'):
    # Cargar modelos y hacer predicciones
    predicciones = []
    for target in targets:
        lower_limit, upper_limit = targets[target]
        model = xgb.Booster()
        model.load_model(f'modelo_xgboost_{target}.json')
        
        dmatrix = xgb.DMatrix(jugador_data)
        y_pred = model.predict(dmatrix, output_margin=True)
        y_pred = np.clip(y_pred, lower_limit, upper_limit)
        intervalo = confidence_interval(y_pred)
        
        predicciones.append({
            'Métrica': target_names[target],
            'Límite Inferior': intervalo[0],
            'Límite Superior': intervalo[1]
        })

    # Convertir lista de predicciones a DataFrame para mostrar como tabla
    predicciones_df = pd.DataFrame(predicciones)
    
    # Mostrar predicciones como tabla
    st.write('Intervalos para el jugador seleccionado y tipo de partido:')
    st.table(predicciones_df)
