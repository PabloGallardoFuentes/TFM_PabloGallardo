import asyncio

from pandas import DataFrame
from carga_datos import cargar_datos


async def obten_media_y_diferencia(df: DataFrame = None) -> DataFrame:
    if df is None:
        df = await cargar_datos(todosLosDatos=False)
    # print(df.head())

    result = DataFrame()
    result["Δcar1²"] = (df.iloc[:, 0] - df.iloc[:, 3])**2
    result["media_car1"] = (df.iloc[:, 0] + df.iloc[:, 3]) / 2
    result["Δcar2²"] = (df.iloc[:, 1] - df.iloc[:, 4])**2
    result["media_car2"] = (df.iloc[:, 1] + df.iloc[:, 4]) / 2
    result["Δcar3²"] = (df.iloc[:, 2] - df.iloc[:, 5])**2
    result["media_car3"] = (df.iloc[:, 2] + df.iloc[:, 5]) / 2

    result["ΔV²"] = df["ΔV"]**2 if "ΔV" in df.columns else df["DV"]**2
    result["ΔECIEDE2000"] = (df["ΔECIEDE2000"] 
                             if "ΔECIEDE2000" in df.columns 
                             else df["DECIEDE2000"] if "DECIEDE2000" in df.columns 
                             else df["DE00"])

    result = result.astype(float)

    print(result.head())
    return result

# df = asyncio.run(obten_media_y_diferencia())
# print("valor de df: ", df)