import pandas as pd
from pathlib import Path

""" Cargamos el csv, cambiamos el nombre de las columnas, ordenamos por
fecha, eliminamos información redundante """
path = Path("datasets/archive/CTA_-_Ridership_-_Daily_Boarding_Totals_20250830.csv")
df = pd.read_csv(path, parse_dates=["service_date"], thousands=",")
df.columns = ["date", "day_type", "bus", "rail", "total"] # shorter names
df = df.sort_values("date").set_index("date")
df = df.drop("total", axis=1) # no need for total, it's just bus + rail
df = df.drop_duplicates() # remove duplicated months (2011-10 and 2014-07)

import matplotlib.pyplot as plt
df["2019-03":"2019-05"].plot(grid=True, marker=".", figsize=(8, 3.5))
plt.show()

""" 
En el plot de una serie que representa los pasajeros que han montado el bus y el rail de
Chicago entre varios meses podemos ver visualmente lo que es una serie de tiempo: información
con valores diferentes en puntos de tiempo diferente, normalmente en intervalos regulares.

En este caso, como hay varios valores por cada punto de tiempo esta es una serie de tiempo 
multivariable (bus y rail).La tarea más típica cuando se tienen series en el tiempo es predecir
valores futuros, aun que también podria utilizarse para rellenar entradas pasadas, clasificación,
detección de anomalias, ...

En estas series podemos ver patrones repetirse, conocidos como "temporadas". Si son muy marcados
estos patrones, con reutilizar muestras pasadas que encajen podría darse una predicción razonable,
a esto se le conoce como predicción naive
"""

""" Representamos naive forecasts"""
diff_7 = df[["bus", "rail"]].diff(7)["2019-03":"2019-05"]
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
df.plot(ax=axs[0], legend=False, marker=".") # original time series
df.shift(7).plot(ax=axs[0], grid=True, legend=False, linestyle=":") # lagged
diff_7.plot(ax=axs[1], grid=True, marker=".") # 7-day difference time series
plt.show()

# Calculamos el error medio absoluto del periodo de 3 meses
print(f"Error medio absoluto { diff_7.abs().mean()}")
""" 
bus 43915.608696
rail 42143.271739
dtype: float64
"""

# Calculamos el porcentage de error medio asoluto
targets = df[["bus", "rail"]]["2019-03":"2019-05"]

print(f"Porcentage del error medio absoluto { (diff_7 / targets).abs().mean()}")
""" 
bus 0.082938
rail 0.089948
dtype: float64 
"""

""" 
Como en la ventana de tres meses que se ha escogido no se puede apreciar una temporalidad evidente
vamos a ampliar la ventana para poder ver años
"""
period = slice("2001", "2019")
df_monthly = df.select_dtypes(include="number").resample('ME').mean()
rolling_average_12_months = df_monthly.loc[period].rolling(window=12).mean()
fig, ax = plt.subplots(figsize=(8, 4))
df_monthly[period].plot(ax=ax, marker=".")
rolling_average_12_months.plot(ax=ax, grid=True, legend=False)
plt.show()

""" 
Se puede observar una temporalidad, sobretodo en el rail, con los mismos picos y valles en los mismos
puntos de tiempo cada año. Ahora vamos a ver la diferencia cada 12 meses
"""

df_monthly.diff(12)[period].plot(grid=True, marker=".", figsize=(8, 3))
plt.show()

""" 
Diferenciando conseguimos eliminar la temporalidad así como las modas, como la que se presentaba de 
2016-2019, en una entrada negativa de personas en los buses y railes. Esta es una tecnica muy comun, ya 
que hace mas sencillo el estudio de estas series; es más facil estudiar una serie temporal "estacionaria" que 
una con modas.

Una vez que se ha entrenado con buenos resultados para la predicción sobre la información diferenciada, es 
sencillo hacer predicciones sobre la información original.
"""

