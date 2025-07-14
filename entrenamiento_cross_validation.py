import asyncio

import numpy as np
import pandas as pd
from sklearn.calibration import cross_val_predict
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from xgboost import XGBRegressor

from graficas import graficar_regresión_lineal_dataframe
from modifica_datos import obten_media_y_diferencia
from stress import stress_df

async def entrena_forest_cv(df: pd.DataFrame):
    input = df.drop(columns=["ΔV²", "ΔECIEDE2000"])
    label = df[["ΔV²", "ΔECIEDE2000"]]
    y = label["ΔV²"]
    eciede2000 = label["ΔECIEDE2000"]

    # Escalado
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input)

    model = RandomForestRegressor(n_estimators=80, random_state=42)

    # KFold para cross-validation (puedes cambiar n_splits)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Predicciones con cross-validation
    y_pred = cross_val_predict(model, input_scaled, y, cv=kf)

    # DataFrame para graficar
    label_df = pd.DataFrame({
        "ΔV": np.sqrt(y),
        "Δpred": np.sqrt(y_pred),
        "ΔECIEDE2000": eciede2000
    })
    await graficar_regresión_lineal_dataframe(label_df, "ΔV", "Δpred", "ΔECIEDE2000")

    # Evaluación
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"RMSE con cross-validation: {rmse}")
    stress = stress_df(np.sqrt(y), np.sqrt(y_pred))
    print(f"Stress: {stress}")


async def entrena_regresion_lineal_cv(df):
    input = df.drop(columns=["ΔV²", "ΔECIEDE2000"])
    label = df[["ΔV²", "ΔECIEDE2000"]]
    y = label["ΔV²"]
    eciede2000 = label["ΔECIEDE2000"]

    # Escalado
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input)

    model = LinearRegression()

    # KFold para cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Predicciones con cross-validation
    y_pred = cross_val_predict(model, input_scaled, y, cv=kf)

    # Dataframe para graficar
    label_df = pd.DataFrame({
        "ΔV": np.sqrt(y),
        "Δpred": np.sqrt(np.where(y_pred < 0, 0, y_pred)),
        "ΔECIEDE2000": eciede2000
    })
    await graficar_regresión_lineal_dataframe(label_df, "ΔV", "Δpred", "ΔECIEDE2000")

    # Evaluación del modelo
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"RMSE regresión lineal con cross-validation: {rmse}")
    stress = stress_df(np.sqrt(y), np.sqrt(np.where(y_pred < 0, 0, y_pred)))
    print(f"Stress: {stress}")

async def entrena_maquina_soporte_vectorial_cv(df):
    input = df.drop(columns=["ΔV²", "ΔECIEDE2000"])
    label = df[["ΔV²", "ΔECIEDE2000"]]
    y = label["ΔV²"]
    eciede2000 = label["ΔECIEDE2000"]

    # Normalización
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    input_scaled = scaler_X.fit_transform(input)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

    # Modelo
    model = SVR(kernel="poly", C=1.0, epsilon=0.1)

    # Cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    y_pred_scaled = cross_val_predict(model, input_scaled, y_scaled, cv=kf)

    # Invertir la normalización de las predicciones
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    # Dataframe para graficar
    label_df = pd.DataFrame({
        "ΔV": np.sqrt(y),
        "Δpred": np.sqrt(np.where(y_pred < 0, 0, y_pred)),
        "ΔECIEDE2000": eciede2000
    })
    await graficar_regresión_lineal_dataframe(label_df, "ΔV", "Δpred", "ΔECIEDE2000")

    # Evaluación del modelo
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"RMSE SVM con cross-validation: {rmse}")
    stress = stress_df(np.sqrt(y), np.sqrt(np.where(y_pred < 0, 0, y_pred)))
    print(f"Stress: {stress}")

    
async def modelo_xgboost_cv(df):
    input = df.drop(columns=["ΔV²", "ΔECIEDE2000"])
    label = df[["ΔV²", "ΔECIEDE2000"]]
    y = label["ΔV²"]
    eciede2000 = label["ΔECIEDE2000"]

    # Definición del modelo con tus hiperparámetros
    model = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )

    # KFold para cross-validation
    kf = KFold(n_splits=7, shuffle=True, random_state=42)

    # Entrenamiento con cross-validation y predicciones
    y_pred = cross_val_predict(model, input, y, cv=kf)

    # Dataframe para graficar
    label_df = pd.DataFrame({
        "ΔV": np.sqrt(y),
        "Δpred": np.sqrt(np.where(y_pred < 0, 0, y_pred)),
        "ΔECIEDE2000": eciede2000
    })
    await graficar_regresión_lineal_dataframe(label_df, "ΔV", "Δpred", "ΔECIEDE2000")

    # Evaluación del modelo
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"RMSE XGBoost con cross-validation: {rmse}")
    stress = stress_df(np.sqrt(y), np.sqrt(np.where(y_pred < 0, 0, y_pred)))
    print(f"Stress XGBoost: {stress}")
    
async def main():

    df = await obten_media_y_diferencia()
    await asyncio.gather(
        entrena_forest_cv(df), 
        # entrena_regresion_lineal_cv(df),
        # entrena_maquina_soporte_vectorial_cv(df),
        # modelo_xgboost_cv(df),
        )
    

asyncio.run(main())