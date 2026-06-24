# Import de la función de separado de conjuntos de entrenamiento/test
from sklearn.model_selection import train_test_split
import torch
def train_test_val(dataset, etiquetas):
    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10
    # train is now 75% of the entire data set
    x_train, x_test, y_train, y_test = train_test_split(dataset, etiquetas, test_size=1 - train_ratio)

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

    return x_train,x_test,x_val,y_train,y_test,y_val
