

import asyncio
import sys
import os

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modifica_datos import obten_media_y_diferencia


async def train_xgboost_model():
    df = await obten_media_y_diferencia()
    X = df[["Δcar1²", "media_car1", "Δcar2²", "media_car2", "Δcar3²", "media_car3"]].values.astype("float32")
    y = df["ΔV²"].values.astype("float32")
    eciede2000 = df["ΔECIEDE2000"].values.astype("float32")
    
    X_train, X_test, y_train, y_test, _, eciede2000_test = train_test_split(
        X, y, eciede2000, test_size=0.1, random_state=42
    )

    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
    )
    model.fit(X_train, y_train)

    return model



if __name__ == "__main__":
    from train_validate_test.entrenamiento_train_validate_test import evaluar_modelo

    model = asyncio.run(train_xgboost_model())
    asyncio.run(evaluar_modelo(model))