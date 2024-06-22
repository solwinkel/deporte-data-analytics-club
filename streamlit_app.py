import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
import json

import os
os.system('pip install openpyxl')

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
train_df = train_df.drop(columns=drop_cols, errors='ignore')
test_df = test_df.drop(columns=drop_cols, errors='ignore')

train_df = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)
test_df = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)

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
    X_test = test_df[features]
    y_test = test_df[target]

    final_model = XGBRegressor(**best_params, random_state=42)
    final_model.fit(X_train, y_train)
    final_model.save_model(f'modelo_xgboost_{target}.json')

# Interfaz de Streamlit
st.title('Predicciones de XGBoost para jugadores')

# Seleccionar un jugador y tipo de partido
jugadores = test_df['jugador'].unique()
tipos_partido = test_df['tipo_partido'].unique()

jugador_seleccionado = st.selectbox('Selecciona un jugador', jugadores)
tipo_partido_seleccionado = st.selectbox('Selecciona un tipo de partido', tipos_partido)

# Filtrar datos del jugador seleccionado
jugador_data = test_df[(test_df['jugador'] == jugador_seleccionado) & 
                       (test_df['tipo_partido'] == tipo_partido_seleccionado)]

# Eliminar columnas no necesarias
jugador_data = jugador_data.drop(columns=['jugador', 'tipo_partido'] + drop_cols + list(targets.keys()), errors='ignore')

# Cargar modelos y hacer predicciones
predicciones = {}
for target in targets:
    lower_limit, upper_limit = targets[target]
    model = xgb.Booster()
    model.load_model(f'modelo_xgboost_{target}.json')
    
    dmatrix = xgb.DMatrix(jugador_data)
    y_pred = model.predict(dmatrix)
    y_pred = np.clip(y_pred, lower_limit, upper_limit)
    predicciones[target] = y_pred[0]

# Mostrar predicciones
st.write('Predicciones para el jugador seleccionado y tipo de partido:')
st.json(predicciones)
