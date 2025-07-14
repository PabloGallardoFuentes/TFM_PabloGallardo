import asyncio

from sklearn.model_selection import train_test_split
import tensorflow as tf


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modifica_datos import obten_media_y_diferencia

async def train_tensorflow_model():
    df = await obten_media_y_diferencia()
    X = df[["Δcar1²", "media_car1", "Δcar2²", "media_car2", "Δcar3²", "media_car3"]].values.astype("float32")
    y = df["ΔV²"].values.astype("float32")
    eciede2000 = df["ΔECIEDE2000"].values.astype("float32")

    X_train, X_test, y_train, y_test, _, eciede2000_test = train_test_split(
        X, y, eciede2000, test_size=0.3, random_state=42
    )
    # Creación del modelo
    modelo = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    modelo.compile(optimizer='adam', loss='mse') #, metrics=[stress_tf])

    modelo.fit(
        X_train, 
        y_train, 
        epochs=100, 
        batch_size=16, 
        validation_split=0.2,
        verbose=1
    )

    return modelo

if __name__ == "__main__":
    from train_validate_test.entrenamiento_train_validate_test import evaluar_modelo

    model = asyncio.run(train_tensorflow_model())

    asyncio.run(evaluar_modelo(model))