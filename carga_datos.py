import pandas as pd


async def cargar_datos(path = './bbdd/COMCorreg-DiffColor.xls', todosLosDatos = True):
    column_names = ["L1", "a1", "b1", "C1", "h1", "L2", "a2", "b2", "C2", "h2", "delete",
                "ΔV", "ΔE CIELAB", "ΔL", "ΔC", "ΔH", "dEuv", 
                "ΔECIE94", "ΔECIEDE2000", "ΔEBFD", "ΔELCD", 
                "ΔECMC", "ΔEThomsen", "ΔEDIN99d", "ΔEVolz", "ΔEGP"]
    df = pd.read_excel(path, skiprows=3, names=column_names, engine="xlrd")
    df = df.drop(columns=["delete"])
    df = df.dropna()
    print(df.shape)
    print(df.head())

    if not todosLosDatos:
        print("1. Seleccionar L, a, b")
        print("2. Seleccionar L, C, h")
        seleccion = int(input("Seleccione una opción: "))
        if seleccion == 1:
            df_final = df[["L1", "a1", "b1", "L2", "a2", "b2", "ΔV", "ΔECIEDE2000"]]
        elif seleccion == 2:
            df_final = df[["L1", "C1", "h1", "L2", "C2", "h2", "ΔV", "ΔECIEDE2000"]]
        else:
            print("Opción no válida")
            exit(1)
        df_final = df_final.dropna()
        print(df_final.shape)

        return df_final
    
    return df

async def cargar_datos_bigc(todosLosDatos = True):
    path = './bbdd/BIGC_Pablo.xlsx'
    column_names = [
        "Pair No.", "n", "Pair", "L1", "a1", "b1", "L2", "a2", "b2", "DV", "DE CIELAB",
        "DV normF1", "DV normF3", "DECIEDE2000", "DV normF1", "DV normF3",
    ]
    df = pd.read_excel(path, skiprows=1, names=column_names, engine="openpyxl")
    df = df.drop(columns=["Pair No.", "n", "Pair", "DE CIELAB", "DV normF1", "DV normF3", "DV normF1", "DV normF3"])
    df = df.dropna()
    print(df.shape)

    if not todosLosDatos:
        df_final = df[["L1", "a1", "b1", "L2", "a2", "b2", "DV", "DECIEDE2000"]]
        df_final = df_final.dropna()
        print(df_final.shape)

        return df_final
    return df
 
async def cargar_datos_icam(todosLosDatos = True):
    path = './bbdd/lcamDIN99WDC_Pablo.xlsx'
    column_names = ["Pair", "L1", "a1", "b1", "L2", "a2", 
                    "b2", "DV", "DV norm1", "DE00", "DV norm2"]

    df = pd.read_excel(path, skiprows=1, names=column_names, engine="openpyxl")
    df = df.drop(columns=["Pair", "DV norm1", "DV norm2",])
    df = df.dropna()
    print(df.shape)

    if not todosLosDatos:
        df_final = df[["L1", "a1", "b1", "L2", "a2", "b2", "DV", "DE00"]]
        df_final = df_final.dropna()
        print(df_final.shape)

        return df_final
    return 

async def cargar_datos_mmb(todosLosDatos = True):
    path = './bbdd/MMB_Pablo.xlsx'
    column_names = ["Center", "Direction", "Pair", "L1", "a1", 
                    "b1", "L2", "a2", "b2", "DV", "DE norm1", "DE00", "DE norm2"]

    df = pd.read_excel(path, skiprows=1, names=column_names, engine="openpyxl")
    df = df.drop(columns=["Center", "Direction", "Pair", "DE norm1", "DE norm2"])
    df = df.dropna()
    print(df.shape)

    if not todosLosDatos:
        df_final = df[["L1", "a1", "b1", "L2", "a2", "b2", "DV", "DE00"]]
        df_final = df_final.dropna()
        print(df_final.shape)

        return df_final
    return df


async def cargar_datos_wanghan(todosLosDatos = True):
    path = './bbdd/WangHan_Pablo.xlsx'
    column_names = ["Pair", "L1", "a1", "b1", "L2", "a2", 
                    "b2", "DV", "DV norm1", "DE00", "DV norm2"]

    df = pd.read_excel(path, skiprows=1, names=column_names, engine="openpyxl")
    df = df.drop(columns=["Pair", "DV norm1", "DV norm2"])
    df = df.dropna()
    print(df.shape)

    if not todosLosDatos:
        df_final = df[["L1", "a1", "b1", "L2", "a2", "b2", "DV", "DE00"]]
        df_final = df_final.dropna()
        print(df_final.shape)

        return df_final
    return df

