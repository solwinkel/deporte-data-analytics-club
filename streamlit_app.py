import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
import os


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

# Interfaz de Streamlit
st.title('Predicciones de XGBoost para jugadores')

# Seleccionar un jugador y tipo de partido
jugadores = test_df['jugador_anonimizado'].unique()
tipos_partido = ['Importante', 'Normal']  # Definimos las categorías directamente

jugador_seleccionado = st.selectbox('Selecciona un jugador', jugadores)
tipo_partido_seleccionado = st.selectbox('Selecciona un tipo de partido', tipos_partido)

# Filtrar datos del jugador seleccionado
jugador_data = test_df[test_df['jugador_anonimizado'] == jugador_seleccionado].copy()

# Eliminar columnas no necesarias
jugador_data = jugador_data.drop(columns=['jugador_anonimizado'] + drop_cols + list(targets.keys()), errors='ignore')

# Agregar la categoría de partido codificada manualmente
if tipo_partido_seleccionado == 'Importante':
    jugador_data['categoria_partido_Importante'] = 1
else:
    jugador_data['categoria_partido_Importante'] = 0

# Asegurarse de que las columnas de jugador_data coincidan con las del modelo
missing_cols = set(X_train.columns) - set(jugador_data.columns)
for col in missing_cols:
    jugador_data[col] = 0

# Reordenar las columnas de jugador_data para que coincidan con X_train
jugador_data = jugador_data[X_train.columns]

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
