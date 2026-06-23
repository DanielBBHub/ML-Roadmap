# Implementacion de un modelo de regresión lineal utilizando tensores y autograd, asi como la API de alto nivel de PyTorch y la aceleración por HW
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

# Definicion de una seed para que el elemento "aleatorio" sea reproducible
torch.manual_seed(42)
# Recogida de la cantidad de caracteristicas
n_features = x_train.shape[1]
# Definicion del valor de los pesos, entre el nº de caracteristicas y 1, tensor con gradiente
w = torch.randn((n_features, 1), requires_grad=True)
# Definición de un escalar a 0, tensor con gradiente
b = torch.tensor(0., requires_grad=True)

# Definición del entrenamiento del modelo
# Definicion de la tasa de aprendizaje
learning_rate = 0.4
# Definicion de las iteraciones de aprendizaje
n_epochs = 20
early_stopping = [0.15,0.0,5.0]
last_loss = 0
for epoch in range(n_epochs):
    # Calculo del valor de la etiqueta predecida como multiplicacion matricial entre
    # la matriz del conjunto de entrenamiento por los pesos y el escalar
    y_pred = x_train @ w + b
    # Calculo de la perdida como el error cuadratico
    loss = ((y_pred - y_train) ** 2).mean()
    # Calculo del gradiente
    loss.backward()
    # Aplicacion del gradiente a los pesos y al escalar
    with torch.no_grad():
        b -= learning_rate * b.grad
        w -= learning_rate * w.grad
    # Reestrablecer el gradiente a 0 despues de la iteración
    b.grad.zero_()
    w.grad.zero_()

    print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item()}")
    if abs(loss - last_loss) < early_stopping[0]:
        if early_stopping[1] == early_stopping[2]:
            print(f"Parada por early stopping con perdida: {loss.item()}")
            break
        else:
            early_stopping[1] += 1
    else:
        last_loss = loss
    

# Una vez entrenado, para hacer inferencia sobre una muestra, se haria de la siguiente manera
# Nuevas muestras
X_new = x_test[:3]
Y_new = y_test[:3]
# no_grad() para no tocar los gradientes entrenados
with torch.no_grad():
    # Prediccion en base a la matriz de caracteristicas multiplicada por los pesos y el escalar
    y_pred = X_new @ w + b

for x,y,pred in zip(X_new,Y_new,y_pred):
    print(f"Para la muestra {x} con etiqueta {y.item():.4f} se ha predecido: {pred.item():.4f}")