import asyncio
import numpy as np
import pandas as pd
import xgboost as xgb

from carga_datos import cargar_datos


def stress_df(delta_V: pd.Series, delta_E: pd.Series) -> float:
    """
    Calcula el estrés de dos dataframes de diferencias de color.
    """
    if delta_V.shape != delta_E.shape:
        raise ValueError("Las series deben tener la misma forma.")
    
    if len(delta_V) == 0 or len(delta_E) == 0:
        return float('nan')
    
    numerator_F1 = np.sum(delta_E**2)
    denominator_F1 = np.sum(delta_E * delta_V)
    if denominator_F1 == 0:
        raise ValueError("El denominador es cero. No se puede calcular el estrés.")
    f1 = numerator_F1 / denominator_F1

    # First definition
    numerator = np.sum((delta_E - f1 * delta_V)**2)
    denominator = np.sum((f1 * delta_V)**2)
    if denominator == 0:
        return 0.0 if numerator == 0 else float('inf')
    stress = np.sqrt(numerator / denominator)

    return stress


def stress_xgboost(y_pred, dtrain):
    delta_V = dtrain
    delta_E = y_pred

    if delta_V.shape != delta_E.shape:
        raise ValueError("Las series deben tener la misma forma.")
    
    numerator_F1 = np.sum(delta_E ** 2)
    denominator_F1 = np.sum(delta_E * delta_V)
    if denominator_F1 == 0:
        return 'stress', float('inf'), True
    
    f1 = numerator_F1 / denominator_F1

    numerator = np.sum((delta_E - f1 * delta_V) ** 2)
    denominator = np.sum((f1 * delta_V) ** 2)

    if denominator == 0:
        stress = 0.0 if numerator == 0 else float('inf')
    else:
        stress = np.sqrt(numerator / denominator)

    return stress



def stress_df_grad_hess(preds: np.ndarray, dtrain: xgb.DMatrix) -> tuple:
    """
    Función personalizada de pérdida para XGBoost que calcula el gradiente
    y hessiano usando el STRESS. Se asume que preds es un array de predicciones.
    """
    y_true = dtrain.get_label()  # Valores reales
    delta_E= preds
    delta_V = y_true

    # Cálculo de F1
    numerator_F1 = np.sum(delta_E**2)
    denominator_F1 = np.sum(delta_E * delta_V)
    if denominator_F1 == 0:
        f1 = 1  # Si el denominador es cero, asignamos un valor por defecto
    else:
        f1 = numerator_F1 / denominator_F1
    
    # Cálculo de gradiente y hessiano (fórmulas de STRESS)
    grad = -2 * (delta_E - f1 * delta_V) * f1 / np.sum((f1 * delta_V)**2)
    hess = 2 * f1 / np.sum((f1 * delta_V)**2)

    return np.full_like(delta_E, grad), np.full_like(delta_E, hess)


# df = asyncio.run(cargar_datos())
# stress_ciede2000 = stress(df["ΔV"], df["ΔECIEDE2000"])
# print(f"Stress: {stress_ciede2000}")