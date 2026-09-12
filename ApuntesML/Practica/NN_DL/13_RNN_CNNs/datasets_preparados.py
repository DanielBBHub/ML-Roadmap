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

# Forecasting Several Time Steps Ahead
""" Hasta ahora solo hemos predicho el valor del siguiente paso temporal, pero
si lo unico que quisieramos fuese predecir un unico valor mas adelante (p.ej.
dentro de 14 dias) bastaria con cambiar el target para que sea el valor 14
pasos despues en vez de 1. La pregunta interesante es: ¿como predecimos los
proximos 14 valores con un unico modelo? Vamos a ver tres tecnicas """

""" 1ª tecnica: reutilizar iterativamente un modelo sequence-to-vector que solo
predice 1 paso (univar_model, entrenado sobre la serie univariable de rail).
Se predice el siguiente valor, se añade a las entradas como si hubiese ocurrido
de verdad, y se repite el proceso tantas veces como pasos queramos predecir """
torch.manual_seed(42)
univar_model = SimpleRnnModel(input_size=1, hidden_size=32, output_size=1)
univar_model = univar_model.to(device)

n_steps = 14
univar_model.eval()
with torch.no_grad():
    # Cogemos los primeros 56 dias (window_length) del periodo de validacion y
    # le añadimos una dimension de lote de tamaño 1 con unsqueeze(), ya que
    # univar_model espera entradas 3D [batch_size, window_length, 1]
    X = rail_valid[:window_length].unsqueeze(dim=0).to(device)
    for step_ahead in range(n_steps):
        y_pred_one = univar_model(X)
        # y_pred_one tiene forma [1, 1]; le añadimos de nuevo una dimension con
        # unsqueeze() para poder concatenarla a X a lo largo del eje temporal
        # (dim=1), como si el valor predicho hubiese ocurrido realmente
        X = torch.cat([X, y_pred_one.unsqueeze(dim=0)], dim=1)
    # Al final X tiene forma [1, 56 + 14, 1]; las predicciones finales son los
    # ultimos 14 valores de X
    Y_pred = X[0, -n_steps:, 0]

""" Si el modelo comete un error en un paso, ese error se arrastra y afecta a
las predicciones de los pasos siguientes: los errores se van acumulando. Por
eso esta tecnica solo es recomendable para predecir un numero pequeño de pasos
hacia el futuro """

""" 2ª tecnica: entrenar una unica RNN para que prediga los 14 valores siguientes
de golpe (sigue siendo sequence-to-vector, pero con output_size=14 en vez de
1). Para ello hace falta cambiar los targets del dataset, de un unico valor a
un vector con los 14 valores siguientes: eso es justo lo que hace
ForecastAheadDataset """
from Forecast_ahead_dataset import ForecastAheadDataset
ahead_train_set = ForecastAheadDataset(mulvar_train, window_length)
ahead_train_loader = DataLoader(ahead_train_set, batch_size=32, shuffle=True)
ahead_eval_set = ForecastAheadDataset(mulvar_valid, window_length)
ahead_eval_loader = DataLoader(ahead_eval_set, batch_size=32, shuffle=True)
ahead_test_set = ForecastAheadDataset(mulvar_test, window_length)
ahead_test_loader = DataLoader(ahead_test_set, batch_size=32, shuffle=True)

# Igual que mulvar_model, pero con output_size=14: predice los 14 valores
# siguientes de rail a partir de la ventana multivariable (rail, bus y tipo de dia)
torch.manual_seed(42)
ahead_model = SimpleRnnModel(input_size=5, hidden_size=32, output_size=14)
ahead_model = ahead_model.to(device)

ahead_model.eval()
with torch.no_grad():
    window = mulvar_valid[:window_length] # shape [56, 5]
    X = window.unsqueeze(dim=0) # shape [1, 56, 5]
    Y_pred = ahead_model(X.to(device)) # shape [1, 14]

""" Esta tecnica funciona bastante bien: las predicciones para el dia siguiente
son mejores que las de 14 dias vista, pero al no reutilizar sus propias
predicciones como entrada, no acumula errores como la primera tecnica. Ademas,
ambas tecnicas se pueden combinar: usar un modelo que predice los siguientes 14
dias de golpe, añadir esas predicciones a las entradas y volver a ejecutar el
modelo para obtener los 14 dias siguientes, y asi sucesivamente (no se puede
usar directamente ahead_model para esto porque necesita tanto rail como bus
como entrada, pero solo predice rail) """

# Forecasting Using a Sequence-to-Sequence Model
""" 3ª tecnica: en vez de entrenar el modelo para que prediga los 14 valores
siguientes solo en el ultimo paso temporal, lo entrenamos para que los prediga
en todos y cada uno de los pasos temporales. En el paso 0 el modelo predice los
pasos 1 a 14, en el paso 1 predice los pasos 2 a 15, etc. El target deja de ser
un vector para pasar a ser una secuencia (de la misma longitud que la entrada)
de vectores de 14 valores. Dada una entrada [batch_size, window_length,
input_size], la salida tendra forma [batch_size, window_length, output_size]:
ya no es un modelo sequence-to-vector, es un modelo sequence-to-sequence
(seq2seq) """
from Seq2seq_dataset import Seq2SeqDataset
seq2seq_train_set = Seq2SeqDataset(mulvar_train, window_length)
seq2seq_train_loader = DataLoader(seq2seq_train_set, batch_size=32, shuffle=True)
seq2seq_eval_set = Seq2SeqDataset(mulvar_valid, window_length)
seq2seq_eval_loader = DataLoader(seq2seq_eval_set, batch_size=32, shuffle=True)
seq2seq_test_set = Seq2SeqDataset(mulvar_test, window_length)
seq2seq_test_loader = DataLoader(seq2seq_test_set, batch_size=32, shuffle=True)

""" Puede parecer trampa que el target contenga valores que tambien aparecen en
la entrada (salvo en el ultimo paso), pero no lo es: en cada paso t una RNN
solo conoce las entradas hasta ese paso, nunca las futuras (es un modelo
causal) """

from seq2seq_model import Seq2SeqRnnModel
torch.manual_seed(42)
seq_model = Seq2SeqRnnModel(input_size=5, hidden_size=32, output_size=14)
seq_model = seq_model.to(device)

seq_model.eval()
with torch.no_grad():
    some_window = mulvar_valid[:window_length] # shape [56, 5]
    X = some_window.unsqueeze(dim=0) # shape [1, 56, 5]
    Y_preds = seq_model(X.to(device)) # shape [1, 56, 14]
    # Durante el entrenamiento se usan las 56 salidas (una por paso temporal),
    # ya que eso aporta muchos mas gradientes de error y estabiliza el
    # entrenamiento; pero para predecir de verdad solo nos interesa la salida
    # del ultimo paso temporal, que es la que resume toda la ventana de entrada
    Y_pred = Y_preds[:, -1] # shape [1, 14]

""" El libro reporta, tras entrenar este modelo, un MAE de validacion de 23.350
para la prediccion a t+1 (muy bueno), que empeora hasta 35.315 para la
prediccion a t+14: es normal que el modelo sea mas preciso cuanto mas cerca
este el horizonte de prediccion.

Las RNN simples funcionan bien prediciendo series de tiempo o manejando
secuencias, pero no tan bien con series o secuencias muy largas """