import asyncio

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modifica_datos import obten_media_y_diferencia
from train_validate_test.entrenamiento_train_validate_test import evaluar_modelo

async def entrena_modelo_xgboost():
    df = await obten_media_y_diferencia()

    input = df.drop(columns=["ΔV²", "ΔECIEDE2000"])
    label = df[["ΔV²", "ΔECIEDE2000"]]


    X_train, X_test, y_train, y_test = train_test_split(
        input, label, test_size=0.1, random_state=42
    )

    y_train = y_train["ΔV²"]
    eciede2000 = y_test["ΔECIEDE2000"]
    y_test = y_test["ΔV²"]

    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
    )
    model.fit(X_train, y_train)

    return model
    


if __name__ == "__main__":

    model = asyncio.run(entrena_modelo_xgboost())

    # Evaluar
    asyncio.run(evaluar_modelo(model))