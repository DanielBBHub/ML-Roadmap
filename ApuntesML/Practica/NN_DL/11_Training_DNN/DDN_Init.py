import torch
import torch.nn as nn

# Por motivos historicos, Linear inicializa los pesos utilizando la uniforme de Kiming, aun que estos estan escalados por un factor de √6
layer = nn.Linear(40, 10)
layer.weight.data *= 6 ** 0.5 # Kaiming init (or 3 ** 0.5 for LeCun init)
torch.zero_(layer.bias.data)

# El metodo anterior funciona, pero es mas limpio utilizar las funciones implementadas en el modulo nn.init
nn.init.kaiming_uniform_(layer.weight)
nn.init.zeros_(layer.bias)

# Si quieres aplicar el mismo método para cada capa del modelo, se puede modificar en el constructor del modelo
# Pero la solución más simple es escribir la siguiente funcion, la cual aplica la inicializacion a los pesos.
def use_he_init(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight)
        nn.init.zeros_(module.bias)

model = nn.Sequential(nn.Linear(50, 40), nn.ReLU(), nn.Linear(40, 1), nn.ReLU())
model.apply(use_he_init)

# El modulo de nn.init tambien contiene un metodo orthogonal_() el cual inicializa los pesos segun una matriz ortogonal:
# Dada una matriz W y una entrada x, la normal de Wx es la misma que la de X, con lo que la magnitud de las entradas
# se conserva en las salidas