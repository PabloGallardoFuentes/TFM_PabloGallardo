import asyncio

from pandas import DataFrame
from carga_datos import cargar_datos


async def obten_media_y_diferencia():
    df = await cargar_datos(todosLosDatos=False)
    # print(df.head())

    result = DataFrame()
    result["Δcar1²"] = (df.iloc[:, 0] - df.iloc[:, 3])**2
    result["media_car1"] = (df.iloc[:, 0] + df.iloc[:, 3]) / 2
    result["Δcar2²"] = (df.iloc[:, 1] - df.iloc[:, 4])**2
    result["media_car2"] = (df.iloc[:, 1] + df.iloc[:, 4]) / 2
    result["Δcar3²"] = (df.iloc[:, 2] - df.iloc[:, 5])**2
    result["media_car3"] = (df.iloc[:, 2] + df.iloc[:, 5]) / 2

    result["ΔV"] = df["ΔV"]
    result["ΔECIEDE2000"] = df["ΔECIEDE2000"]

    print(result.head())
    return result

df = asyncio.run(obten_media_y_diferencia())
print("valor de df: ", df)