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

from Dataset_serie_temporal import TimeSeriesDataset

# Comprobacion funcionamiento de TimeSeriesDataset()
my_series = torch.tensor([[0], [1], [2], [3], [4], [5]])
my_dataset = TimeSeriesDataset(my_series, window_length=3)
for window, target in my_dataset:
    print("Window:", window, " Target:", target)

from torch.utils.data import DataLoader
# Creacion dataloader del Dataset
torch.manual_seed(0)
my_loader = DataLoader(my_dataset, batch_size=2, shuffle=True)

for X, y in my_loader:
    print("X:", X, " y:", y)
""" X: tensor([[[0], [1], [2]], [[2], [3], [4]]]) y: tensor([[3], [5]])
X: tensor([[[1], [2], [3]]]) y: tensor([[4]])

Como podemos ver, cuando el tamaño del lote no es un multiplo de la longitud
del lote, el ultimo lote sera mas corto.

Habiendo comprobado que podemos transfromar una serie temporal en un dataset con el
que entrenar un modelo de ML prepararemos el dataset real."""

""" Dividiremos la serie temporal en 3, entrenamiento, validacion y test y convertiremos los
datos numericos a una precision de floats de 32 bits, seguido de un escalado por un factor de
un millon para asegurarnos que esten dentro de un rango [0,1] """
rail_train = torch.FloatTensor(df[["rail"]]["2016-01":"2018-12"].values / 1e6)
rail_valid = torch.FloatTensor(df[["rail"]]["2019-01":"2019-05"].values / 1e6)
rail_test = torch.FloatTensor(df[["rail"]]["2019-06":].values / 1e6)


window_length = 56
train_set = TimeSeriesDataset(rail_train, window_length)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
valid_set = TimeSeriesDataset(rail_valid, window_length)
valid_loader = DataLoader(valid_set, batch_size=32)
test_set = TimeSeriesDataset(rail_test, window_length)
test_loader = DataLoader(test_set, batch_size=32)

# Forecasting Multivariate Time Series
""" Una de las cualidades mas importantes de las redes neuronales es su flexibilidad; podemos
manejar series temporales multivariadas sin tener que cambiar casi nada de la arquitectura """

# Crearmos un df multivariable escalado
df_mulvar = df[["rail", "bus"]] / 1e6 
# Definimos el dia siguiente para tenerlo en cuenta en el entrenamiento
df_mulvar["next_day_type"] = df["day_type"].shift(-1) 
# Codificamos el tipo de dia a float
df_mulvar = pd.get_dummies(df_mulvar, dtype=float) 

mulvar_train = torch.FloatTensor(df_mulvar["2016-01":"2018-12"].values / 1e6)
mulvar_valid = torch.FloatTensor(df_mulvar["2019-01":"2019-05"].values / 1e6)
mulvar_test = torch.FloatTensor(df_mulvar["2019-06":].values / 1e6)

mulvar_train_set = MulvarTimeSeriesDataset(mulvar_train, window_length)
mulvar_train_loader = DataLoader(mulvar_train_set, batch_size=32, shuffle=True)
mulvar_eval_set = MulvarTimeSeriesDataset(mulvar_valid, window_length)
mulvar_eval_loader = DataLoader(mulvar_eval_set, batch_size=32, shuffle=True)
mulvar_test_set = MulvarTimeSeriesDataset(mulvar_test, window_length)
mulvar_test_loader = DataLoader(mulvar_test_set, batch_size=32, shuffle=True)

from modelo_rnn import SimpleRnnModel
torch.manual_seed(42)
mulvar_model = SimpleRnnModel(input_size=5, hidden_size=32, output_size=1)
mulvar_model = mulvar_model.to(device)