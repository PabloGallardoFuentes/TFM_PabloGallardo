import asyncio
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from skimage.color import lab2rgb

from carga_datos import cargar_datos


async def graficar_regresión_lineal_dataframe(df, columna_X, columna_Y, eciede2000):
    plt.figure(figsize=(8, 5))
    sns.regplot(data=df, x=columna_X, y=columna_Y,  
                scatter_kws={"alpha": 0.5}, line_kws={"color": "blue", "label": f"Regresión modelo"},
                label=f"Predicciones modelo") 
                

    reg2 = sns.regplot(data=df, x=columna_X, y=eciede2000,  
                scatter_kws={"alpha": 0.5}, line_kws={"color": "red", "label": "Regresión CIEDE2000"},
                label="Predicciones CIEDE2000")

    # Calcular límites comunes
    x_vals = df[columna_X]
    y_vals = df[columna_Y]
    min_val = min(x_vals.min(), y_vals.min())
    max_val = max(x_vals.max(), y_vals.max())

    # Dibujar línea x = y
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='x = y') 

    # Ajustar ejes
    # plt.axis("equal")
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)

    plt.title(f"{columna_X} vs {columna_Y}")
    plt.xlabel(columna_X)
    plt.ylabel("ΔE")
    plt.legend()
    plt.show()

async def regresion_lineal(x, y):
    """
    Realiza una regresión lineal entre dos ndarrays y grafica los resultados.
    
    :param x: ndarray con los valores de la variable independiente.
    :param y: ndarray con los valores de la variable dependiente.
    """
    
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, alpha=0.5, label="Datos reales")
    plt.plot(x, y, color="red", label="Regresión lineal")
    plt.title("Regresión Lineal")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.legend()
    plt.show()

    

async def main():
    df = await cargar_datos(todosLosDatos=True)
    print(df.head())

    # Graficar regresión lineal
    # await graficar_regresión_lineal_dataframe(df, "ΔV", "ΔECIEDE2000")

    # TODO: graficar con la media de los dos puntos
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    df = df.astype(float)
    L = df["L1"].to_numpy()
    a = df["a1"].to_numpy()
    b = df["b1"].to_numpy()
    lab_colors = np.column_stack((L, a, b))  # Normalización de valores Lab
    rgb_colors = lab2rgb(lab_colors.reshape(1, -1, 3)).reshape(-1, 3)  # Conversión a RGB

    # Convertir valores Lab a RGB (aproximación)
    # norm = lambda v: (v - v.min()) / (v.max() - v.min())  # Normalización simple
    # colors = np.array([norm(x), norm(y), norm(z)]).T  # RGB normalizado

    # Graficar con color
    ax.scatter(a, b, L, c=rgb_colors, marker='o', s=100)
    ax.set_xlabel('L')
    ax.set_ylabel('a')
    ax.set_zlabel('b')

    plt.show()


# asyncio.run(main())