# Implementando capas de agrupación (pooling) en pytorch

import torch
from torch import torch.nn as nn

# MaxPool (tamaño del kernel y opcionalmente el padding)
max_pool = nn.MaxPool2d(kernel_size=2)

# AvgPool
max_pool = nn.AvgPool2d(kernel_size=2)

# Actualmente se utiliza más maxpool, ya que trae un cálculo más rápido, menos uso de memória, conserva
# las características más importantes y ofrece más invarianza, teniendo en cuenta que "pierdes" más información
# También se puede aplicar estas capas en profundidad, en vez de en el espacio, lo cual permite a la red aprender
# a ser invariante ante varias características, asegurando que la salida es la misma aun habiendo traslación, un valor
# de brillo mas alto, un color diferente ...
# Pytorch no cuenta con una implementación de pooling en profundidad, pero la siguiente sería un modulo personalizado
# bsaado en F.max_pool1d()

import torch.nn.functional as F
class DepthPool(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        
    def forward(self, inputs):
        batch, channels, height, width = inputs.shape
        Z = inputs.view(batch, channels, height * width) # merge spatial dims
        Z = Z.permute(0, 2, 1) # switch spatial and channels dims
        Z = F.max_pool1d(Z, kernel_size=self.kernel_size, stride=self.stride,
        padding=self.padding) # compute max pool
        Z = Z.permute(0, 2, 1) # switch back spatial and channels dims
        return Z.view(batch, -1, height, width) # unmerge spatial dims

# GlobalAvgPool
# Esta capa funciona de manera bastante diferente, calcula la media de cada mapa de características, con lo que la salida es
# un único número por cada característica e instancia. Aún que destruye casi toda la información, puede ser útil para utilizarla
# antes de la capa de salida

# Para implementarla, se puede crear una capa de agrupación avg 2d y definir el tamaño de salida como 1  
global_avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
output = global_avg_pool(cropped_images)

# También se puede utilizar la función .mean()
output = cropped_images.mean(dim=(2, 3), keepdim=True)