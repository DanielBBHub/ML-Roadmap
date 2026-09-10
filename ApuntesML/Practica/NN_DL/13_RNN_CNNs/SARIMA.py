import pandas as pd
from pathlib import Path

""" Aplicaremos el modelo SARIMA al conjunto de datos de rail para predecir
los pasajeros para el primer dia junio 2019"""

path = Path("datasets/archive/CTA_-_Ridership_-_Daily_Boarding_Totals_20250830.csv")
df = pd.read_csv(path, parse_dates=["service_date"], thousands=",")
df.columns = ["date", "day_type", "bus", "rail", "total"] # shorter names
df = df.sort_values("date").set_index("date")
df = df.drop("total", axis=1) # no need for total, it's just bus + rail
df = df.drop_duplicates() # remove duplicated months (2011-10 and 2014-07)

""" Podemos descargar el modelo ARIMA de la libreria statsmodel """
from statsmodels.tsa.arima.model import ARIMA
# Obtenemos los datos del rail desde principios de 2019 hasta finales de mayo 2019
# y definimos asfreq("D") para declarar una frecuencia diaria
origin, today = "2019-01-01", "2019-05-31"
rail_series = df.loc[origin:today]["rail"].asfreq("D")

""" Creamos un modelo ARIMA con la información hasta el final de mayo y definimos los
hiperparametros como order=(1, 0, 0), lo cual equivale a p = 1, d = 0 y q = 0
y seasonal_order=(0, 1, 1, 7) que equivale a P = 0, D = 1, Q = 1 y s = 7"""
model = ARIMA(rail_series,
order=(1, 0, 0),
seasonal_order=(0, 1, 1, 7))
# Ejecutamos el fit() del modelo para poder obtener una prediccion
model = model.fit()
y_pred = model.forecast() 
print(f"Predicción pasajeros rail para 1 junio 2019: {y_pred["2019-06-01"]:.0f}")
# Predicción pasajeros rail para 1 junio 2019: 427759

""" La predicción ha sido de 427759 mientras que el dato real fueron 379044, una desviación
del 12.9%, poco peor que la predicción naiv de 426932 (12.6%). Para comprobar
que no fuese mala suerte podemos hacer una prediccion para cada dia de los meses
de marzo, abril y mayo y calcular MAE para ese periodo"""

origin, start_date, end_date = "2019-01-01", "2019-03-01", "2019-05-31"
time_period = pd.date_range(start_date, end_date)
rail_series = df.loc[origin:end_date]["rail"].asfreq("D")
y_preds = []

for today in time_period.shift(-1):
    model = ARIMA(rail_series[origin:today], # train on data up to "today"
    order=(1, 0, 0),
    seasonal_order=(0, 1, 1, 7))
    model = model.fit() # note that we retrain the model every day!
    y_pred = model.forecast().iloc[0]
    y_preds.append(y_pred)

y_preds = pd.Series(y_preds, index=time_period)
mae = (y_preds - rail_series[time_period]).abs().mean() # returns 32,040.7
print(f"MAE para el periodo de marzo, abril y mayo: {mae:.2f}")