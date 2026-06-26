## IMPLEMENTING LINEAR REGRESSION

import torch.nn as nn
import torch
# Import del dataset de info de inmuebles de california
from sklearn.datasets import fetch_california_housing
from ModelUtl.Train import train_bgd, train_test_val

housing = fetch_california_housing()


x_train, x_test, x_val, y_train, y_test, y_val = train_test_val(housing.data, housing.target)

n_features = x_train.shape[1]
torch.manual_seed(42) 
# Definicion del modelo lineal con importacion de la API de alto nivel
# Tantas neuronas de entrada como caracteristicas, una unica neurona de salida
model = nn.Linear(in_features=n_features, out_features=1)

# El modulo del modelo de regresion lineal tiene tanto un tensor con "subjetividad", tantos valores como neuronas de salida,
# como un tensor con los pesos del modelo.
# Estos parametros son instancias de la clase Parameter, hija de la clase Tensor
print(f"\nSubjetividad del modelo: {model.bias}")
print(f"\nPesos del modelo: {model.weight}")

# Podemos acceder a todos los parametros de un modelo mediante el siguiente codigo, no devuelve Tensors,
# solo la subclase Parameter
i = 0
for param in model.parameters():
    print(f"\nParametro numero {i}: {param}")
    i+= 1

for x,y in zip(x_train[:2],y_train[:2]):
    print(f"\nPrediccion del modelo sin entrenar: {model(x)} ||||||||| Etiqueta real: {y}")

# Ya teniendo el modelo elegido, escogemos un optimizador y una funcion de perdida
# Escogemos el Stochastic Gradient Descent
learning_rate = 0.4
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# La clase MSELoss tambien es un modulo, asi que dandole las predicciones y las etiquetas calculara el error
mse = nn.MSELoss()

train_bgd(model, optimizer, mse, x_train, y_train, 30)

# Una vez entrenado, para hacer inferencia sobre una muestra, se haria de la siguiente manera
# Nuevas muestras
X_new = x_test[:3]
Y_new = y_test[:3]
# no_grad() para no tocar los gradientes entrenados
with torch.no_grad():
    # Prediccion en base a la matriz de caracteristicas multiplicada por los pesos y el escalar
    y_pred = model(X_new)

for x,y,pred in zip(X_new,Y_new,y_pred):
    print(f"Para la muestra {x} con etiqueta {y.item():.4f} se ha predecido: {pred.item():.4f}")