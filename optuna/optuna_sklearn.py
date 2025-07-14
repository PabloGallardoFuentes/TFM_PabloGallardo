import asyncio
from math import inf
import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import SVR
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modifica_datos import obten_media_y_diferencia
from stress import stress_df

X_train = None
X_test = None
delta_V_train = None
delta_V_test = None
input = None
label = None

async def obten_datos():
    df = await obten_media_y_diferencia()
    global X_train, X_test, delta_V_train, delta_V_test, input, label

    input = df.drop(columns=["ΔV²", "ΔECIEDE2000"])
    label = df["ΔV²"]
    # X_train, X_test, delta_V_train, delta_V_test = train_test_split(input, label, test_size=0.2, random_state=42)
    # return X_train, X_test, delta_V_train, 

def objective(trial):
    model_type = trial.suggest_categorical("model", ["SVR"])
    if model_type == "SVR":
        C = trial.suggest_float("svr_c", 1e-3, 1e3, log=True)
        epsilon = trial.suggest_float("epsilon", 1e-3, 1, log=True)
        model = SVR(C=C, epsilon=epsilon)
        return inf
    else:
        max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        n_estimators = trial.suggest_int("rf_n_estimators", 10, 100, step=5)
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)

    # División simple del conjunto en entrenamiento y test
    X_train, X_test, y_train, y_test = train_test_split(
        input, label, test_size=0.3, random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = pd.Series(model.predict(X_test))

    stress = stress_df(y_test, y_pred)

    return stress
    # kf = KFold(n_splits=8, shuffle=True, random_state=42)
    # stresses = []
    # for train_index, test_index in kf.split(input):
    #     X_train, X_test = input.iloc[train_index], input.iloc[test_index]
    #     delta_V_train, delta_V_test = label.iloc[train_index], label.iloc[test_index]

    #     model.fit(X_train, delta_V_train)
    #     delta_V_pred = pd.Series(model.predict(X_test))

    #     stress = stress_df(delta_V_test, delta_V_pred)
    #     stresses.append(stress)

    # return np.mean(stresses)

async def main():
    await obten_datos()
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, show_progress_bar=True, n_jobs=4)

    print("Best trial: ", study.best_trial)
    print("Best params: ", study.best_trial.params)

asyncio.run(main())