# Diferentes implementaciones de diferentes optimizadores para acelerar el entrenamiento 
# redes neuronales

# Momentum optimizer (definir el parametro momentum. 0.9 suele funcionar bien)
optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=0.05)

# NGA (definir el parametro nesterov=True)
optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, nesterov=True, lr=0.05)

# AdaGrad (Mejor utilizarlo en tareas de regresion lineal)
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)

# RMSProp (definir el parametro de desintegración alpha)
optimizer = torch.optim.RMSprop(model.parameters(), alpha=0.9, lr=0.05)

# Adam (el parametro de momentum se suele definir a 0.9 y el de desintegración a 0.999)
optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=0.05)

## Variantes de Adam 
optimizer = torch.optim.NAdam(model.parameters(), betas=(0.9, 0.999), lr=0.05)
optimizer = torch.optim.Adamax(model.parameters(), betas=(0.9, 0.999), lr=0.05)
# Para AdamW propablemente haya que modificar el hiperparametro de desintegración de pesos
optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.999), lr=0.05)

