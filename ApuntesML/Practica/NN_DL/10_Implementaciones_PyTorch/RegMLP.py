## IMPLEMENTING A REGRESSION MLP WITH HIGH LVL API

import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
from LinearRegAPI import train_bgd
from ModelUtl.Train import train_test_val

x_train, x_test, x_val, y_train, y_test, y_val = train_test_val(housing.data, housing.target)

torch.manual_seed(42)
n_features = x_train.shape[1]
# Pytorch tiene un modulo Sequential que utiliza ejecuta de manera secuencial la predicción sobre los inputs
# y utiliza la salida de estos como entrada de los siguientes [Input -> Linear -> ReLU -> Linear -> Relu -> Linear -> Pred]
model = nn.Sequential(
# Capa de entrada con n neuronas de entrada; tantas como caracteristicas en los inputs y 50 salidas
nn.Linear(n_features, 50),
# Funcion de activacion para la primera capa intermedia
nn.ReLU(),
# Tantas entradas como salidas en el modelo anterior y n salidas
nn.Linear(50, 40),
# Funcion de activacion para la primera capa intermedia
nn.ReLU(),
# Tantas entradas como salidas en el modelo anterior y 1 salida para la prediccion
nn.Linear(40, 1)
)

learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
mse = nn.MSELoss()
train_bgd(model, optimizer, mse, x_train, y_train, 35)

# Una vez entrenado, para hacer inferencia sobre una muestra, se haria de la siguiente manera
# Nuevas muestras
X_new = x_test[:3]
Y_new = y_test[:3]
# no_grad() para no tocar los gradientes entrenados
with torch.no_grad():
    # Prediccion en base a la matriz de caracteristicas multiplicada por los pesos y el escalar
    y_pred = model(X_new)

for x,y,pred in zip(X_new,Y_new,y_pred):
    print(f"Para la muestra {x} con etiqueta {y.item():.4f} se ha predecido: {pred.item():.4f} |||||||| Diferencia de: {abs(y.item() - pred.item()):.4f}")