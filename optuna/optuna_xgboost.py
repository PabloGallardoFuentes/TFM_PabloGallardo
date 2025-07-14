import asyncio
import numpy as np
import optuna
import sys
import os
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from carga_datos import cargar_datos
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
    # label = np.sqrt(df["ΔV²"])
    label = df["ΔV²"]
    # df = await cargar_datos(todosLosDatos=True)
    # df = df.astype(float)

    # input = df[["L1", "a1", "b1", "C1", "h1", "L2", "a2", "b2", "C2", "h2"]]
    # label = df["ΔV"]
    # print(input.head())
    X_train, X_test, delta_V_train, delta_V_test = train_test_split(input, label, test_size=0.1, random_state=42)
    # return X_train, X_test, delta_V_train, 

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 1.0, log=True)
    }

    kf = KFold(n_splits=7, shuffle=True, random_state=42)
    stresses = []
    for train_index, test_index in kf.split(X_train):
        X_train_fold, X_test_fold = input.iloc[train_index], input.iloc[test_index]
        delta_V_train_fold, delta_V_test_fold = label.iloc[train_index], label.iloc[test_index]
        model = XGBRegressor(**params, random_state=42, verbosity=0)
        model.fit(X_train_fold, delta_V_train_fold)

        preds = model.predict(X_test_fold)
        stress_error = stress_df(pd.Series(delta_V_test_fold), pd.Series(preds))
        stresses.append(stress_error)

    # model = XGBRegressor(**params, random_state=42, verbosity=0)
    # model.fit(X_train, delta_V_train)

    # preds = model.predict(X_test)
    # stress_error = stress_df(pd.Series(delta_V_test), pd.Series(preds))

    return np.mean(stresses)

async def main():
    await obten_datos()
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    print("Best trial: ", study.best_trial)
    print("Best params: ", study.best_trial.params)

asyncio.run(main())