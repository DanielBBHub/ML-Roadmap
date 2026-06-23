import torch.nn as nn
import torch
# Import del dataset de info de inmuebles de california
from sklearn.datasets import fetch_california_housing
# Import de la función de separado de conjuntos de entrenamiento/test
from sklearn.model_selection import train_test_split

housing = fetch_california_housing()
train_ratio = 0.75
validation_ratio = 0.15
test_ratio = 0.10

# train is now 75% of the entire data set
x_train, x_test, y_train, y_test = train_test_split(housing.data, housing.target, test_size=1 - train_ratio)

# test is now 10% of the initial data set
# validation is now 15% of the initial data set
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 



# Definicion de un tensor con el conjunto de entrenamiento
x_train = torch.FloatTensor(x_train)
# Definicion de un tensor con el conjunto de validacion
x_val = torch.FloatTensor(x_val)
# Definicion de un tensor con el conjunto de test
x_test = torch.FloatTensor(x_test)

# Transformaciones de los conjuntos ----------
# Se obtienen las medias
means = x_train.mean(dim=0, keepdims=True)
# Se obtienen las desviaciones stf
stds = x_train.std(dim=0, keepdims=True)
# Se normalizan los valores restandole las medias y dividiendo entre las desviaciones, 
# calculadas sobre el conjunto de entrenamiento
x_train = (x_train - means) / stds
x_val = (x_val - means) / stds
x_test = (x_test - means) / stds

# Convertimos los arrays de etiquetas en tensores, ya que las predicciones seran vectores-columna 
# y los arrays de numpy son vectores unidimensionales, con lo que las redimensionamos añadiendo una dimension
y_train = torch.FloatTensor(y_train).reshape(-1, 1)
y_val = torch.FloatTensor(y_val).reshape(-1, 1)
y_test = torch.FloatTensor(y_test).reshape(-1, 1)

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

# El bucle de entrenamiento es igual, pero ahora ya no se trabaja con tensores y autograd directamente,
# si no que los modulos se encargan de hacer ese trabajo
def train_bgd(model, optimizer, criterion, X_train, y_train, n_epochs):
    for epoch in range(n_epochs):
        y_pred = model(X_train)
        # Calculo de la perdida con el modulo recibido
        loss = criterion(y_pred, y_train)
        # Calculo de gradiente de pesos y escalar
        loss.backward()
        # Momento de actualizar pesos y escalar; descenso de gradiente
        optimizer.step()
        # Reestablecer valores a 0 para el gradiente
        optimizer.zero_grad()
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item()}")

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