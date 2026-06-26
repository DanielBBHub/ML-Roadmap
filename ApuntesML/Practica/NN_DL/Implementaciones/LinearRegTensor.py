## IMPLEMENTING A REGRESSION MLP WITH TENSORS
import torch
# Import del dataset de info de inmuebles de california
from sklearn.datasets import fetch_california_housing
# Import de la función de separado de conjuntos de entrenamiento/test
from sklearn.model_selection import train_test_split

from ModelUtl.Train import train_test_val


housing = fetch_california_housing()
# train is now 75% of the entire data set
x_train, x_test, x_val, y_train, y_test, y_val = train_test_val(housing.data, housing.target)

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