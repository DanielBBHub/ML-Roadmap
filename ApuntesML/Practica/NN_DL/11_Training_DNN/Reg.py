# Diferentes formas de implementar regularizacion en el entrenamiento de las redes
# neuronales profundas

# Regularizacion l2
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
# Definicion de la lista de parametros a regularizar en la red. Se evita tocar los pesos de los 
# sesgos, así como los de batch-norm
params_to_regularize = [param for name, param in model.named_parameters()
if not "bias" in name and not "bn" in name]
for epoch in range(n_epochs):
    for X_batch, y_batch in train_loader:
        [...] # bucle de entrenamiento
        # Calculo de la funcion de perdida
        main_loss = loss_fn(y_pred, y_batch)
        # Calulo de la perdida l2
        l2_loss = sum(param.pow(2.0).sum() for param in params_to_regularize)
        # Escalado de la perdida por l2
        loss = main_loss + 1e-4 * l2_loss

# Otra forma de hacerlo es utilizar la funcionalidad de PyTorch de "parameter groups" para aplicar diferentes hiperparametros
# con el optimizador. Hasta ahora hemos creado los optimizadores con todos los parametros del modelo, los cuales van todos
# al mismo grupo de parametros, pero podemos pasar una lista de diccionarios con una entrada "params". Tambien se pueden
# añadir entradas para algunos hiperparametros especificos para este grupo.

## Creamos una lista con los parametros de sesgo y batch-norm
params_bias_and_bn = [param for name, param in model.named_parameters()
                      if "bias" in name or "bn" in name]
# Le pasamos al optimizador una lista de diccionarios; dos entradas "params" con los parametros, la primera con una
# propiedad "weight_decay" para aplicarlo a los primeros parametros. Una vez definido el optimizador de esta manera
# podria entrenarse de manera normal.                      
optimizer = torch.optim.SGD([
    {"params": params_to_regularize, "weight_decay": 1e-4, "lr": 0.05},
    {"params": params_bias_and_bn, "lr": 0.01},  
], lr=0.05)

# Merece la pena destacar que los grupos de parametros permiten aplicar distintos valores de tasa de aprendizaje,
# normalmente utilizado en transfer learning para actualizar más rápido las capas nuevas

# Regularización l1
# PyTorch no tiene ninguna implementacion de la regularizacion l1, con lo que hay que implementarla manualmente
params_to_regularize = [param for name, param in model.named_parameters()
if not "bias" in name and not "bn" in name]
for epoch in range(n_epochs):
    for X_batch, y_batch in train_loader:
        [...] # bucle de entrenamiento
        # Calculo de la funcion de perdida
        main_loss = loss_fn(y_pred, y_batch)
        # Calulo de la perdida l1
        l1_loss = sum(param.abs().sum() for param in params_to_regularize)        
        # Escalado de la perdida por l1
        loss = main_loss + 1e-4 * l1_loss

# Dropout
# Para implementar dropout en un modelo basta con definid una instancia de nn.Dropout para cada capa intermedia. 
# Esta solo funcionara cuando model.train() este activado
model = nn.Sequential(
nn.Flatten(),
nn.Dropout(p=0.2), nn.Linear(1 * 28 * 28, 100), nn.ReLU(),
nn.Dropout(p=0.2), nn.Linear(100, 100), nn.ReLU(),
nn.Dropout(p=0.2), nn.Linear(100, 100), nn.ReLU(),
nn.Dropout(p=0.2), nn.Linear(100, 10)
).to(device)

# Es importante destacar que comparar los resultados de la pérdida en el entrenamiento y los resultados de la 
# evaluación del modelo será engañoso ya que el primero estará afectado por el dropout.

# Por otro lado, si en vez de utilizar ReLU se implementa un modelo con una función de activación basada en SELU
# habria que cambiar a alpha dropout, ya que esta si guardaría la media y varianza de sus entradas

# Dropout Monte Carlo
# Definimos el modo evaluacion antes de hacer las predicciones
model.eval()
# Pero iteramos todas las capas de la red neuronal para poner en modo entrenamiento aquellas que sean dropout
for module in model.modules():
    if isinstance(module, nn.Dropout):
        module.train()

X_new = X[:3] # some new images, e.g., the first 3 images of the test set
X_new = X_new.to(device)
torch.manual_seed(42)

with torch.no_grad():
    # Generamos un lote de 100 imagenes de la misma imagen con repeat_interleave(), con forma [300, 1, 28, 28] 
    # ya que estan todas en la primera dimension
    X_new_repeated = X_new.repeat_interleave(100, dim=0)
    # El modelo genera predicciones con 10 logits por imagen con una forma de tensor [300, 10],
    # el cual reestructuramos a [3, 100, 10] para agrupar las predicciones para cada imagen
    y_logits_all = model(X_new_repeated).reshape(3, 100, 10)
    # Transformamos los logits en probabilidades con la funcion softmax
    y_probas_all = torch.nn.functional.softmax(y_logits_all, dim=-1)
    # Calculamos la media de la segunda dimension para obtener la pobabilidad media estimada
    # de cada una de las clases en cada una de las imagenes con resultado de un tensor [3, 10]
    y_probas = y_probas_all.mean(dim=1)

# Regularizacion Max-Norm
def apply_max_norm(model, max_norm=2, epsilon=1e-8, dim=1):
    with torch.no_grad():
        for name, param in model.named_parameters():
            # Se itera unicamente sobre los pesos de la red
            if 'bias' not in name:
                # Se calcula la normalizacion l2
                actual_norm = param.norm(p=2, dim=dim, keepdim=True)
                # Se calcula la normalizacion de la etiqueta para el peso de cada neurona,
                # creando una copia del tensor actual_norm en la que se reemplazan todos los
                # valores > max_norm por max_norm, por r en la ecuacion de Max-Norm
                target_norm = torch.clamp(actual_norm, 0, max_norm)
                # Reescalamos la matriz con los pesos para que cada columna acabe con la 
                # normalización de la etiqueta
                param *= target_norm / (epsilon + actual_norm)