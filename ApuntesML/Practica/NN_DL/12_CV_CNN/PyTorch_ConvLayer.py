import numpy as np
import torch
from sklearn.datasets import load_sample_images

sample_images = np.stack(load_sample_images()["images"])
# Las imagenes son u-bytes con lo que las convertimos a float32 y normalizamos valores de 0-255 a 0-1
sample_images = torch.tensor(sample_images, dtype=torch.float32) / 255

print(f"Forma del tensor con las imágenes: {sample_images.shape}")
# Forma del tensor con las imágenes: torch.Size([2, 427, 640, 3])

# Ya que pytorch espera tener una entrada con la forma [canales, alto, ancho] tenemos que permutar el orden
# de las imagenes en el tensor

sample_images_permuted = sample_images.permute(0, 3, 1, 2)

import torchvision
import torchvision.transforms.v2 as T

cropped_images = T.CenterCrop((70, 120))(sample_images_permuted)
print(f"Forma del tensor con las imágenes croppeadas: {cropped_images.shape}")
# Forma del tensor con las imágenes croppeadas: torch.Size([2, 3, 70, 120])

import torch.nn as nn

torch.manual_seed(42)
# Creación de una red convolucional 
conv_layer = nn.Conv2d(
    # 3 canales de entrada
    in_channels=3, 
    # 32 filtros
    out_channels=32, 
    # kernel/filtro de tamaño 7x7
    kernel_size=7)
fmaps = conv_layer(cropped_images)

print(f"Forma del tensor con las imagenes de salida (padding valid): {fmaps.shape}")
# Forma del tensor con las imagenes de salida: torch.Size([2, 32, 64, 114])
# Como hemos definido 32 canales de salida, ahora ademas de los canales RGB en cada punto
# tenemos la intensidad de cada característica en cada punto
# Por otro lado, ha encogido la imagen 6px debido a que la capa 2d no utiliza cero padding con lo 
# que se pierden pixeles en los lados de los mapas de características (como es una matriz 7x7 se pierden)
# 6px en horizontal y 6px en vertical (3 px por cada lado)

# Por defecto el hiperparametro de padding es 0, tambien conocido como valid padding, ya que el campo receptivo
# de cada neurona recae estrictamente dentro de "posiciones válidas" de la entrada. Esto es equivalente a padding="valid"

conv_layer_padd = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7,
padding="same")
fmaps2 = conv_layer_padd(cropped_images)
print(f"Forma del tensor con las imagenes de salida (padding same): {fmaps2.shape}")
# Forma del tensor con las imagenes de salida (padding same): torch.Size([2, 32, 70, 120])
# En este caso, padding="same" fuerza a que la imagen de salida tenga las mismas dimensiones que la de entrada, poniendo
# 0s en los puntos necesarios

print(f"Forma de los parametros de la capa. \nPesos: {conv_layer.weight.shape} \nBias: {conv_layer.bias.shape}")
# Forma de los parametros de la capa. 
# Pesos: torch.Size([32, 3, 7, 7]) 
# Bias: torch.Size([32])
# El tensor de pesos es 4D y su forma es: [output_channels, input_channels, kernel_height, kernel_width]
# El tensor de bias es 1D con: [output_channels]
# Ambos parametros se inicializan de manera aleatoria utilizando una distribución +-1/(k⁻²), donde k = fh × fw × fn.
# De la misma manera que en las redes profundas, dependiendo de la función de activación, habrá que utilizar una inicializacion
# en concreto
# Puedes utilizar cualquier imagen de entrada mientras sea igual de grande o mas que el kernel de filtros.
# Es impoertante utilizar funciones de activación despues de cada capa convolucional por la misma razon que en las 
# redes lineales, para poder "aprender" patrones complejos

