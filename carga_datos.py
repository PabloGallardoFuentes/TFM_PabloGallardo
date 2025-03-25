import pandas as pd


async def cargar_datos(path = './COMCorreg-DiffColor.xls', todosLosDatos = True):
    column_names = ["L1", "a1", "b1", "C1", "h1", "L2", "a2", "b2", "C2", "h2", "delete",
                "ΔV", "ΔE CIELAB", "ΔL", "ΔC", "ΔH", "dEuv", 
                "ΔECIE94", "ΔECIEDE2000", "ΔEBFD", "ΔELCD", 
                "ΔECMC", "ΔEThomsen", "ΔEDIN99d", "ΔEVolz", "ΔEGP"]
    df = pd.read_excel(path, skiprows=3, names=column_names, engine="xlrd")
    df = df.drop(columns=["delete"])
    print(df.shape)
    # print(df.head())

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