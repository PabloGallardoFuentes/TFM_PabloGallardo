import asyncio
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from tabulate import tabulate

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from carga_datos import cargar_datos_bigc, cargar_datos_icam, cargar_datos_mmb, cargar_datos_wanghan
from entrenamiento_train_validate_test_pytorch import MLPRegressor
from graficas import graficar_regresión_lineal_dataframe
from modifica_datos import obten_media_y_diferencia
from stress import stress_df

async def compara_stress_por_grupos(label_df: pd.DataFrame):
    # División en grupos
    group_labels = ['<1', '1-5', '>5']
    conditions = [
        label_df["ΔV"] < 1,
        (label_df["ΔV"] >= 1) & (label_df["ΔV"] <= 5),
        label_df["ΔV"] > 5
    ]

    stress_results = []
    resultados = {}
    for label, condition in zip(group_labels, conditions):
        group_df = label_df[condition]
        stress_val = stress_df(group_df["ΔV"], group_df["Δpred"])
        stress_eciede_val = stress_df(group_df["ΔV"], group_df["ΔECIEDE2000"])
        print(f"  Grupo {label}:")
        print(f"  Stress: {stress_val}")
        print(f"  Stress ECIEDE2000: {stress_eciede_val}")
        stress_results.append((label, stress_val, stress_eciede_val))
        resultados[label] = (stress_val, stress_eciede_val)
        
    # Gráfico de barras
    labels = [r[0] for r in stress_results]
    stress_vals = [r[1] for r in stress_results]
    stress_eciede_vals = [r[2] for r in stress_results]

    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar([i - width/2 for i in x], stress_vals, width, label='Stress')
    ax.bar([i + width/2 for i in x], stress_eciede_vals, width, label='Stress ECIEDE2000')

    ax.set_ylabel('Stress')
    ax.set_title('Stress vs Stress ECIEDE2000 por Grupo de ΔV')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    plt.show()

    return resultados


async def _print_stress_metrics(name: str, label_df: pd.DataFrame):
    print(f"\n====== {name} ======")
    stress = stress_df(label_df["ΔV"], label_df["Δpred"])
    print(f"Stress: {stress}")
    stress_eciede2000 = stress_df(label_df["ΔV"], label_df["ΔECIEDE2000"])
    print(f"Stress ECIEDE2000: {stress_eciede2000}")
    print("========================\n")
    await graficar_regresión_lineal_dataframe(
        label_df, "ΔV", "Δpred", "ΔECIEDE2000"
    )

    resultados_por_grupos = await compara_stress_por_grupos(label_df)
    return resultados_por_grupos

async def _muestra_tabla_resultados(tabla_dict: dict):
    def _format_tuple(t):
        if isinstance(t, tuple) and all(pd.notna(x) for x in t):
            return f"{t[0]:.4f} / {t[1]:.4f}"
        else:
            return "(nan, nan)"
        
    tabla_df = pd.DataFrame.from_dict(tabla_dict, orient='index')
    # print(tabla_df.head())
    # print(tabla_df.columns.to_list())
    tabla_df = tabla_df[["<1", "1-5", ">5"]]  # ordena las columnas si quieres

    # Formato
    tabla_df_str = tabla_df.copy()
    new_columns = {}
    for col in tabla_df_str.columns:
        tabla_df_str[col] = tabla_df_str[col].apply(_format_tuple)
        new_columns[col] = f"{col} (Stress modelo / Stress ECIEDE2000)"

    tabla_df_str.rename(columns=new_columns, inplace=True)
    tabla_formateada = tabulate(
        tabla_df_str,
        headers="keys",
        tablefmt="grid",
        showindex=True,
    )
    print("\n", tabla_formateada)
    # tabla_mostrable = tabla_df.applymap(lambda val: f"{val[0]:.3f} / {val[1]:.3f}" if val[0] != "nan" else "NaN / NaN")
    # print(tabla_mostrable.to_string())

async def entrena_modelo_forest_regressor() -> RandomForestRegressor:
    df = await obten_media_y_diferencia()
    
    input = df.drop(columns=["ΔV²", "ΔECIEDE2000"])
    label = df[["ΔV²", "ΔECIEDE2000"]]

    X_train, X_test, y_train, y_test = train_test_split(
        input, label, test_size=0.3, random_state=42
    )
    y_train = y_train["ΔV²"]
    eciede2000 = y_test["ΔECIEDE2000"]
    y_test = y_test["ΔV²"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(
        max_depth=13,
        n_estimators=75, 
        random_state=42)
    model.fit(X_train, y_train)

    return model

async def evaluar_modelo(model: RandomForestRegressor | MLPRegressor):
    tabla_resultados_por_bbdd = {}
    ## BIGC
    raw_df = await cargar_datos_bigc(False) 
    df = await obten_media_y_diferencia(raw_df)

    input = df.drop(columns=["ΔV²", "ΔECIEDE2000"])
    label = df[["ΔV²", "ΔECIEDE2000"]]

    y_pred = model.predict(input)
    print(y_pred)
    label_df =pd.DataFrame({
        "ΔV": np.sqrt(label["ΔV²"]),
        "Δpred": np.sqrt(y_pred.ravel() if y_pred.ndim > 1 else y_pred),
        "ΔECIEDE2000": label["ΔECIEDE2000"]
    })
    tabla_resultados_por_bbdd['BIGC'] = await _print_stress_metrics("Base de datos BIGC", label_df)


    ## ICAM
    raw_df = await cargar_datos_icam(False)
    df = await obten_media_y_diferencia(raw_df)

    input = df.drop(columns=["ΔV²", "ΔECIEDE2000"])
    label = df[["ΔV²", "ΔECIEDE2000"]]

    y_pred = model.predict(input)
    label_df = pd.DataFrame({
        "ΔV": np.sqrt(label["ΔV²"]),
        "Δpred": np.sqrt(y_pred.ravel() if y_pred.ndim > 1 else y_pred),
        "ΔECIEDE2000": label["ΔECIEDE2000"]
    })
    tabla_resultados_por_bbdd['ICAM'] = await _print_stress_metrics("Base de datos ICAM", label_df)


    ## MMB
    raw_df = await cargar_datos_mmb(False)
    df = await obten_media_y_diferencia(raw_df)

    input = df.drop(columns=["ΔV²", "ΔECIEDE2000"])
    label = df[["ΔV²", "ΔECIEDE2000"]]
    y_pred = model.predict(input)

    label_df = pd.DataFrame({
        "ΔV": np.sqrt(label["ΔV²"]),
        "Δpred": np.sqrt(y_pred.ravel() if y_pred.ndim > 1 else y_pred),
        "ΔECIEDE2000": label["ΔECIEDE2000"]
    })
    tabla_resultados_por_bbdd['MMB'] = await _print_stress_metrics("Base de datos MMB", label_df)


    # WangHan
    raw_df = await cargar_datos_wanghan(False)
    df = await obten_media_y_diferencia(raw_df)

    input = df.drop(columns=["ΔV²", "ΔECIEDE2000"])
    label = df[["ΔV²", "ΔECIEDE2000"]]
    y_pred = model.predict(input)

    label_df = pd.DataFrame({
        "ΔV": np.sqrt(label["ΔV²"]),
        "Δpred": np.sqrt(y_pred.ravel() if y_pred.ndim > 1 else y_pred),
        "ΔECIEDE2000": label["ΔECIEDE2000"]
    })
    tabla_resultados_por_bbdd['WangHan'] = await _print_stress_metrics("Base de datos WangHan", label_df)

    await _muestra_tabla_resultados(tabla_resultados_por_bbdd)

if __name__ == "__main__":

    model = asyncio.run(entrena_modelo_forest_regressor())

    # Evaluar
    asyncio.run(evaluar_modelo(model))