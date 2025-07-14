import asyncio
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from xgboost import XGBRegressor

from carga_datos import cargar_datos
from graficas import graficar_regresión_lineal_dataframe
from modifica_datos import obten_media_y_diferencia
from stress import stress_df, stress_xgboost

async def entrena_forest(df: pd.DataFrame):
    input = df.drop(columns=["ΔV²", "ΔECIEDE2000"])
    label = df[["ΔV²", "ΔECIEDE2000"]]

    X_train, X_test, y_train, y_test = train_test_split(
        input, label, test_size=0.1, random_state=42
    )
    y_train = y_train["ΔV²"]
    eciede2000 = y_test["ΔECIEDE2000"]
    y_test = y_test["ΔV²"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=80, random_state=42)
    model.fit(X_train, y_train)

    # Predicciones
    y_pred = model.predict(X_test)

    
    # Dataframe para graficar
    label_df = pd.DataFrame({"ΔV": np.sqrt(y_test), "Δpred": np.sqrt(y_pred), "ΔECIEDE2000": eciede2000})
    await graficar_regresión_lineal_dataframe(label_df, "ΔV", "Δpred", "ΔECIEDE2000")

    # Evaluación del modelo
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE random forest: {rmse}")
    stress = stress_df(np.sqrt(y_test), np.sqrt(y_pred))
    print(f"Stress: {stress}")

async def entrena_regresion_lineal(df):
    # Cargar los datos
    # df = await cargar_datos(todosLosDatos=False)
    # df = await obten_media_y_diferencia(df)

    input = df.drop(columns=["ΔV²", "ΔECIEDE2000"])
    label = df[["ΔV²", "ΔECIEDE2000"]]

    X_train, X_test, y_train, y_test = train_test_split(
        input, label, test_size=0.1, random_state=42
    )
    y_train = y_train["ΔV²"]
    eciede2000 = y_test["ΔECIEDE2000"]
    y_test = y_test["ΔV²"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predicciones
    y_pred = model.predict(X_test)

    # Dataframe para graficar
    label_df = pd.DataFrame({"ΔV²": np.sqrt(y_test), "Δpred": np.sqrt(y_pred), "ΔECIEDE2000": eciede2000})
    await graficar_regresión_lineal_dataframe(label_df, "ΔV²", "Δpred", "ΔECIEDE2000")

    # Evaluación del modelo
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE regresion lineal: {rmse}")
    stress = stress_df(np.sqrt(y_test), np.sqrt(y_pred))
    print(f"Stress: {stress}")

async def entrena_maquina_soporte_vectorial(df):
    # df = await cargar_datos(todosLosDatos=False)
    # df = await obten_media_y_diferencia(df)

    input = df.drop(columns=["ΔV²", "ΔECIEDE2000"])
    label = df[["ΔV²", "ΔECIEDE2000"]]

    X_train, X_test, y_train, y_test = train_test_split(
        input, label, test_size=0.1, random_state=42
    )
    y_train = y_train["ΔV²"]
    eciede2000 = y_test["ΔECIEDE2000"]
    y_test = y_test["ΔV²"]

    # Normalización
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

    # Entrenamiento del modelo
    model = SVR(kernel="poly", C=1.0, epsilon=0.1)
    model.fit(X_train, y_train)

    #Predicciones
    y_pred = model.predict(X_test)
    # y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    # Dataframe para graficar
    label_df = pd.DataFrame({"ΔV": np.sqrt(y_test), "Δpred": np.sqrt(y_pred), "ΔECIEDE2000": eciede2000})
    await graficar_regresión_lineal_dataframe(label_df, "ΔV", "Δpred", "ΔECIEDE2000")

    # Evaluación del modelo
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE SVM: {rmse}")
    stress = stress_df(y_test, y_pred)
    print(f"Stress: {stress}")


async def modelo_xgboost(df):
    input = df.drop(columns=["ΔV²", "ΔECIEDE2000"])
    label = df[["ΔV²", "ΔECIEDE2000"]]
    X_train, X_test, y_train, y_test = train_test_split(
        input, label, test_size=0.15, random_state=42
    )
    y_train = y_train["ΔV²"]
    eciede2000 = y_test["ΔECIEDE2000"]
    y_test = y_test["ΔV²"]

    params = {
        "n_estimators": 55,
        "max_depth": 3,
        "learning_rate": 0.0001672414771262658,
        "subsample": 0.8372014516412533,
        "colsample_bytree": 0.8086588961471524,
        "random_state": 42,
        "reg_alpha": 0.1920984404860121,
        "reg_lambda": 0.05572840361621161,
    }

    # grid_search.fit(X_train, y_train)
    # print("Mejores parámetros: ", grid_search.best_params_)
    # return
    model = XGBRegressor(
        n_estimators= 100,
        max_depth= 5,
        learning_rate= 0.1,
        subsample= 0.8,
        colsample_bytree= 0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

    y_pred = model.predict(X_test)
    
    label_df = pd.DataFrame({"ΔV": np.sqrt(y_test), "Δpred": np.sqrt(y_pred), "ΔECIEDE2000": eciede2000})
    await graficar_regresión_lineal_dataframe(label_df, "ΔV", "Δpred", "ΔECIEDE2000")

    # Evaluación del modelo
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE xgboost: {rmse}")
    stress = stress_df(y_test, y_pred)
    print(f"Stress xgboost: {stress}")
    stress_eciede2000 = stress_df(y_test, eciede2000)
    print(f"Stress ECIEDE2000: {stress_eciede2000}")

    

async def main():

    df = await obten_media_y_diferencia()
    await asyncio.gather(
        # entrena_forest(df), 
        # entrena_regresion_lineal(df),
        # entrena_maquina_soporte_vectorial(df),
        modelo_xgboost(df)
        # investigar sobre si se puede pasar stress a la hora de entrenar
        # mirar modelos de red neuronal
        )
    

asyncio.run(main())
