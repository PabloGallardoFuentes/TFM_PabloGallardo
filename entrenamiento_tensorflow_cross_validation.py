import asyncio
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import tensorflow as tf

from graficas import graficar_regresión_lineal_dataframe
from modifica_datos import obten_media_y_diferencia

def stress_tf(y_true, y_pred):
    numerator_F1 = tf.reduce_sum(tf.square(y_pred))
    denominator_F1 = tf.reduce_sum(y_pred * y_true)
    f1 = numerator_F1 / denominator_F1

    numerator = tf.reduce_sum(tf.square(y_pred - f1 * y_true))
    denominator = tf.reduce_sum(tf.square(f1 * y_true))

    stress = tf.sqrt(numerator / denominator)
    return stress

def crear_modelo(input_dim):
    return tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

async def main():
    df = await obten_media_y_diferencia()

    X = df[["Δcar1²", "media_car1", "Δcar2²", "media_car2", "Δcar3²", "media_car3"]].values.astype("float32")
    y = df[["ΔV²", "ΔECIEDE2000"]].values.astype("float32")
    eciede2000 = df["ΔECIEDE2000"].values.astype("float32")
    y = df["ΔV²"].values.astype("float32")

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    rmses = []
    stresses = []

    mejor_stress = float('inf')
    mejor_resultado_df = None

    for i, (train_idx, test_idx) in enumerate(kf.split(X_scaled)):
        print(f"\nFold {i+1}")

        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_scaled[train_idx], y_scaled[test_idx]

        modelo = crear_modelo(X.shape[1])
        modelo.compile(optimizer='adam', loss='mse', metrics=[stress_tf])

        modelo.fit(
            X_train,
            y_train,
            epochs=100,
            batch_size=16,
            validation_split=0.2,
            verbose=0
        )

        y_pred_scaled = modelo.predict(X_test).ravel()
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

        rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
        stress = stress_tf(tf.constant(y_test_orig, dtype=tf.float32),
                           tf.constant(y_pred, dtype=tf.float32)).numpy()

        print(f"Fold {i+1} RMSE: {rmse}")
        print(f"Fold {i+1} Stress: {stress}")

        rmses.append(rmse)
        stresses.append(stress)

        if stress < mejor_stress:
            mejor_stress = stress
            mejor_resultado_df = pd.DataFrame({
                "ΔV": np.sqrt(y_test_orig),
                "Δpred": np.sqrt(y_pred),
                "ΔECIEDE2000": eciede2000[test_idx]
            })

    print(f"\nMean RMSE: {np.mean(rmses)}")
    print(f"Mean Stress: {np.mean(stresses)}")

    await graficar_regresión_lineal_dataframe(mejor_resultado_df, "ΔV", "Δpred", "ΔECIEDE2000")

asyncio.run(main())
