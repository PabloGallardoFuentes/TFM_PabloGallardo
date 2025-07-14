import asyncio
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import tensorflow as tf

from modifica_datos import obten_media_y_diferencia

def stress_tf(y_true, y_pred):
    """
    Calcula el estrés de dos tensores de diferencias de color.
    """
    # if tf.shape(y_true) != tf.shape(y_pred):
    #     raise ValueError("Los tensores deben tener la misma forma.")
    
    numerator_F1 = tf.reduce_sum(tf.square(y_pred))
    denominator_F1 = tf.reduce_sum(y_pred * y_true)
    f1 = numerator_F1 / denominator_F1

    # First definition
    numerator = tf.reduce_sum(tf.square(y_pred - f1 * y_true))
    denominator = tf.reduce_sum(tf.square(f1 * y_true))
    
    stress = tf.sqrt(numerator / denominator)

    return stress

async def main():
    df = await obten_media_y_diferencia()

    X = df[["Δcar1²", "media_car1", "Δcar2²", "media_car2", "Δcar3²", "media_car3"]].values.astype("float32")
    y = (df["ΔV²"]).values.astype("float32")

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # Creación del modelo
    modelo = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    modelo.compile(optimizer='adam', loss='mse', metrics=[stress_tf])

    modelo.fit(
        X_train, 
        y_train, 
        epochs=100, 
        batch_size=16, 
        validation_split=0.2,
        verbose=1
    )

    y_pred_scaled = modelo.predict(X_test).ravel()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

    # Evaluacion
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
    print(f"RMSE: {rmse}")
    stress = stress_tf(tf.constant(y_test_orig, dtype=tf.float32), 
                       tf.constant(y_pred, dtype=tf.float32)).numpy()
    print(f"Stress: {stress}")






asyncio.run(main())

