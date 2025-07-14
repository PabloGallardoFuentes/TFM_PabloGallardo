import asyncio
import keras
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import layers
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modifica_datos import obten_media_y_diferencia
from stress import stress_df


X_train = None
X_test = None
delta_V_train = None
delta_V_test = None

async def obten_datos():
    df = await obten_media_y_diferencia()

    input = df.drop(columns=["ΔV²", "ΔECIEDE2000"])
    # label = np.sqrt(df["ΔV²"])
    label = df["ΔV²"]
    global X_train, X_test, delta_V_train, delta_V_test
    X_train, X_test, delta_V_train, delta_V_test = train_test_split(input, label, test_size=0.3, random_state=42)
    # return X_train, X_test, delta_V_train, 


def objective(trial):
    n_layers = trial.suggest_int("n_layers", 1, 7)
    hidden_units = trial.suggest_int("hidden_units", 8, 128, log=True)
    activation = trial.suggest_categorical("activation", ["relu", "tanh", "selu"])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    epochs = trial.suggest_int("epochs", 10, 500)

    #Crear el modelo
    model = keras.Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))
    for _ in range(n_layers):
        model.add(layers.Dense(hidden_units, activation=activation))
    model.add(layers.Dense(1))

    # Compilar el modelo
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="mse")
    # Entrenar
    model.fit(X_train, delta_V_train, batch_size=batch_size, epochs=epochs, verbose=0)

    # Prediccion y stress
    delta_V_pred = model.predict(X_test).flatten()
    return stress_df(pd.Series(delta_V_test), pd.Series(delta_V_pred))

async def main():
    await obten_datos()
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=400, show_progress_bar=True)

    print("Best trial: ", study.best_trial)
    print("Best params: ", study.best_trial.params)

asyncio.run(main())