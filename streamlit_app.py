import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy import stats

train_df = pd.read_excel('train_df.xlsx')
test_df = pd.read_excel('test_df.xlsx')

train_df2 = train_df
test_df2 = test_df

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

train_df = train_df.drop(columns=drop_cols, errors='ignore')
test_df = test_df.drop(columns=drop_cols, errors='ignore')

train_df = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)
test_df = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)

missing_cols = set(train_df.columns) - set(test_df.columns)
for col in missing_cols:
    if col not in targets:
        test_df[col] = 0

test_df = test_df[train_df.drop(columns=targets.keys()).columns]

best_params = {
    'colsample_bytree': 0.8,
    'learning_rate': 0.1,
    'max_depth': 3,
    'min_child_weight': 3,
    'n_estimators': 50,
    'subsample': 0.6
}

for target in targets:
    lower_limit, upper_limit = targets[target]
    features = [col for col in train_df.columns if col != target and col not in targets]

    X_train = train_df[features]
    y_train = train_df[target]
#     final_model = XGBRegressor(**best_params, random_state=42)
#     final_model.fit(X_train, y_train)
#     final_model.save_model(f'modelo_xgboost_{target}.json')

def confidence_interval(predictions, confidence=0.99, n_bootstraps=1000):
    bootstrapped_means = []
    for _ in range(n_bootstraps):
        samples = np.random.choice(predictions, size=len(predictions), replace=True)
        bootstrapped_means.append(np.mean(samples))
    
    lower_bound = np.percentile(bootstrapped_means, (1 - confidence) / 2 * 100)
    upper_bound = np.percentile(bootstrapped_means, (1 + confidence) / 2 * 100)
    
    if(round(lower_bound,1) == round(upper_bound,1)):
        lower_bound = lower_bound - lower_bound * 0.1
        upper_bound = upper_bound + upper_bound * 0.1

    return lower_bound, upper_bound

target_names = {
    'minutos': 'Minutos Jugados',
    'avg_dist_sess_m': 'Distancia Promedio (m)',
    'zona_4_19.9_25.1_kmh': 'Zona 4 (19.9-25.1 km/h)',
    'zona_5_mas_25.1_kmh': 'Zona 5 (>25.1 km/h)',
    'num_aceleraciones_intensas': 'Aceleraciones Intensas',
    'num_desaceleraciones_intensas': 'Desaceleraciones Intensas',
    'num_acel_desintensas': 'Aceleraciones + Desaceleraciones Intensas',
    'num_sprints_total': 'Sprints Totales',
    'prom_esfuerzos_repetidos': 'Esfuerzos Repetidos',
    'max_vel_kmh': 'Velocidad Máxima (km/h)'
}

#armado de la interfaz
st.title('Predicciones de métricas físicas para jugadores 🏋️')

jugadores = test_df['jugador_anonimizado'].unique()
tipos_partido = ['Importante', 'Normal']  

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="column">', unsafe_allow_html=True)
    jugador_seleccionado = st.selectbox('Selecciona un jugador', jugadores)
    tipo_partido_seleccionado = st.selectbox('Selecciona un tipo de partido', tipos_partido)
    st.markdown('</div>', unsafe_allow_html=True)

jugador_data = test_df[test_df['jugador_anonimizado'] == jugador_seleccionado].copy()

info_jugador = train_df[train_df['jugador_anonimizado'] == jugador_seleccionado].iloc[0]

with col2:
    st.markdown('<div class="column">', unsafe_allow_html=True)
    st.write(f"Edad 🎂: {info_jugador['edad']}")
    st.write(f"Altura 🧍↕: {info_jugador['altura']} cm")
    st.write(f"Peso ⏲️: {info_jugador['peso']} kg")

    posicion_habitual = train_df2[train_df2['jugador_anonimizado'] == jugador_seleccionado]['posicion_habitual'].iloc[0]
    st.write(f"Posición habitual 🏅: {posicion_habitual}")

    num_partidos_train = len(train_df[train_df['jugador_anonimizado'] == jugador_seleccionado])
    num_partidos_test = len(test_df[test_df['jugador_anonimizado'] == jugador_seleccionado])
    num_partidos = num_partidos_train + num_partidos_test
    st.write(f"Cantidad de partidos ⚽: {num_partidos}")

    num_mas_55_train = len(train_df2[(train_df2['jugador_anonimizado'] == jugador_seleccionado) & (train_df2['minutos'] > 50)])
    num_mas_55_test = len(test_df2[(test_df2['jugador_anonimizado'] == jugador_seleccionado) & (test_df2['minutos'] > 50)])
    num_mas_55 = num_mas_55_train + num_mas_55_test
    st.write(f"Cantidad de partidos jugados de titular ⏱️: {num_mas_55}")
    st.markdown('</div>', unsafe_allow_html=True)


jugador_data = jugador_data.drop(columns=['jugador_anonimizado'] + drop_cols + list(targets.keys()), errors='ignore')

if tipo_partido_seleccionado == 'Normal':
    jugador_data['categoria_partido_Normal'] = 1
else:
    jugador_data['categoria_partido_Normal'] = 0

missing_cols = set(train_df.drop(columns=targets.keys()).columns) - set(jugador_data.columns)
for col in missing_cols:
    jugador_data[col] = 0

jugador_data = jugador_data[train_df.drop(columns=targets.keys()).columns]

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
            'Límite Inferior': max(intervalo[0], lower_limit), 
            'Límite Superior': min(intervalo[1], upper_limit) 
        })

    predicciones_df = pd.DataFrame(predicciones)
    
    st.write('Intervalos para el jugador seleccionado y tipo de partido:')
    st.table(predicciones_df)
