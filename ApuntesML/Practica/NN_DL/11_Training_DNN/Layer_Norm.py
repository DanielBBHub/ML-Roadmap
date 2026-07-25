

# PyTorch incluye un modulo nn.LayerNorm en el que solo tienes uqe indicar las dimensiones que quieres normalizar.
# Estas tienen que ser las dimensiones de la entrada, en el caso de una imagen a color 100x200, la forma sería [3, 100, 200]

# Este codigo crea un lote de imágenes a color aleatorias y una LN que
# normalizara las dos ultimas dimensiones
inputs = torch.randn(32, 3, 100, 200) 
layer_norm = nn.LayerNorm([100, 200])
result = layer_norm(inputs) 

# Normalmente en aplicaciones de visión artificial se suele normalizar todas 
# las dimensiones a la vez, con lo que se puede añadir la dimension del canal
# para que normalice la entrada completa
layer_norm = nn.LayerNorm([3, 100, 200])
result = layer_norm(inputs)