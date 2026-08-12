from functools import partial
import torch
from torch import torch.nn as nn

# Una posible implementación de una red convolucional para la predicción de clases en el dataset de FashionMNIST
# Encadena capas convolucionales, ReLU y capas de pooling para darle una salida que alimentará una red neuronal directa

# Utilizamos .partial() para definir DefaultConv2d, que actuara igual que Conv2d pero teniendo argumentos de entrada diferentes
# para evitar tener que repetir el tamaño del kernel y el padding
DefaultConv2d = partial(nn.Conv2d, kernel_size=3, padding="same")
model = nn.Sequential(
    # La primera capa es DefaultConv2d con 64 canales de salida y un kernel relativamente grande (7x7)
    # ademas de un solo canal de entrada ya que el dataset de FashionMNIST solo tiene imagenes en grayscale
    # seguida de una ReLU, como el resto de capas convolucionales
    DefaultConv2d(in_channels=1, out_channels=64, kernel_size=7), nn.ReLU(),
    # Definimos una capa de Pooling que reducira las dimensiones en 2
    nn.MaxPool2d(kernel_size=2),
    # Es importante resaltar que los canales van duplicandose entre capas convolucionales, ya que puede que las caracteristicas
    # de bajo nivel sean pocas, pero son muchas las posibles combinaciones entre estas
    DefaultConv2d(in_channels=64, out_channels=128), nn.ReLU(),
    DefaultConv2d(in_channels=128, out_channels=128), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    DefaultConv2d(in_channels=128, out_channels=256), nn.ReLU(),
    DefaultConv2d(in_channels=256, out_channels=256), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    # Finalmente, la red neuronal directa, compuesta por una capa flatten, para cambiar la forma de la entrada
    # de 2D a 1D, seguida de dos capas Linear con ReLU para acabar en una capa densa que sacará logits para cada posible clase
    # Para evitar el overfitting se añade además dos capas dropout
    nn.Flatten(),
    nn.Linear(in_features=2304, out_features=128), nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(in_features=128, out_features=64), nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(in_features=64, out_features=10),
    ).to(device)