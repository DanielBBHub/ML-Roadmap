# PyTorch incluye modulos para diferentes funciones de activacion de DDNN como nn.LeakyReLU, RReLU o PReLU. Es recomendable utilizarlas con la
# inicialización de Kamming
import torch
import torch.nn as nn

# Implementación de un modelo con Leaky ReLU como función de activación
alpha = 0.2
model = nn.Sequential(nn.Linear(50, 40), nn.LeakyReLU(negative_slope=alpha))
nn.init.kaiming_uniform_(model[0].weight, alpha, nonlinearity="leaky_relu")

# Implementación de un modelo con ELU como función de activación
alpha2 = 0.2
model2 = nn.Sequential(nn.Linear(50, 40), nn.ELU(alpha=alpha2))
nn.init.kaiming_uniform_(model[0].weight, alpha, nonlinearity="leaky_relu")