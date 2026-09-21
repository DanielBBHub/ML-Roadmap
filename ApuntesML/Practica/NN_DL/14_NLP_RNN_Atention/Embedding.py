import torch.nn as nn
""" 
Pytorch ofrece el modulo nn.Embedding, el cual envuelve una matriz de incrustación. Esta tiene una fila por cada posible categoría y una columna por cada 
dimensión de embedding. Esta dimensionalidad es un hiperparametro configurable, aun que por defecto se inicializa aleatoriamente
"""

# Convertir una categoría de ID a un embedding. La capa de embedding mira y devuelve la fila correspondiente
torch.manual_seed(42)
embed = nn.Embedding(5, 3) # 5 categories × 3D embeddings
print(f"Categorias embedidas: {embed(torch.tensor([[3, 2], [0, 2]]))}")

""" 
tensor([[[ 0.2674, 0.5349, 0.8094],
[ 2.2082, -0.6380, 0.4617]],

[[ 0.3367, 0.1288, 0.2345],
[ 2.2082, -0.6380, 0.4617]]], grad_fn=<EmbeddingBackward0>)

Aqui podemos ver como la categoría 3 queda incrustada como el vector 3D [ 0.2674, 0.5349, 0.8094], la categoría 2 queda incrustada 2 veces como [ 2.2082, -0.6380, 0.4617] y
la 0 como [ 0.3367, 0.1288, 0.2345]
"""