import asyncio
import matplotlib.pyplot as plt
import seaborn as sns

from carga_datos import cargar_datos


async def main():
    df = await cargar_datos(todosLosDatos=True)

    plt.figure(figsize=(8, 5))
    sns.regplot(data=df, x="ΔV", y="ΔECIEDE2000",  scatter_kws={"alpha": 0.5}, line_kws={"color": "red"})

    plt.title("ΔV vs ΔECIEDE2000")
    plt.xlabel("ΔV")
    plt.ylabel("ΔECIEDE2000")
    plt.show()


asyncio.run(main())