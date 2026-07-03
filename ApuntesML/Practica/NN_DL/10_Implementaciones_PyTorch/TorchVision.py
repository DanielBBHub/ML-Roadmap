# BUILDING AN IMAGE CLASSIIER WITH TORCHVISION

# La libreria de TorchVision es una parte grande del ecosistema de Pytorch, implementa
# muchas herramientas de vision artificial, tiene una implementacion de herramientas como
#   - Descarga de datasets
#   - Modelos preentrenados
#   - Transformacion de imagenes


import torchvision
import torchvision.transforms.v2 as T
import torchmetrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# Comprobacion en tiempo de ejecucion para el soporte de CUDA
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Objeto que se encargara de transformar el formato original del conjunto (PIL) a floats
# En este caso es un "Compose" que encadena dos transformaciones: PIL -> Image (Clase de TorchVision) -> Float
toTensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

# Caracteristica de datasets de TorchVision:
#   - Aceptan un argumento transform, el cual aplicaran a todas las muestras
#   - Se puede aplicar el target_transform si se quiere modificar tambien las etiquetas
# Por otro lado, el resto de argumentos definen los siguiente:
#   - Root: carpeta donde se guardara el conjunto
#   - Train: indicador de carga del conjunto de entrenamiento
#   - Download: indicador de descarga del conjunto
train_and_valid_data = torchvision.datasets.FashionMNIST(
root="datasets", train=True, download=True, transform=toTensor)

test_data = torchvision.datasets.FashionMNIST(
root="datasets", train=False, download=True, transform=toTensor)

torch.manual_seed(42)
learning_rate = 0.002 

train_data, valid_data = torch.utils.data.random_split(
train_and_valid_data, [55_000, 5_000])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
eval_loader = DataLoader(valid_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

# En la primera implementacion de un modelo para el Fashion MNIST la entrada era un array de 784 valores de intensidad,
# ahora la entrada es un tensor de 3 dimensiones ([1,28,28]):
#   - La dimension del canal (para ByN solo 1 canal)
#   - Ancho
#   - Alto

from ImgClass import ImageClassifier
from ModelUtl.Train import train_minibatch_gd

torch.manual_seed(42)
model = ImageClassifier(n_inputs=28 * 28, n_hidden1=300, n_hidden2=100,
n_classes=10)
model = model.to(device)
# Como es un modelo multiclase, se utiliza CrossEntropyLoss, ya que acepta cualquier clase como
# una prediccion o como probabilidades
xentropy = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
train_minibatch_gd(model, optimizer, xentropy, train_loader, eval_loader, 30, device, accuracy)

model.eval()
X_new, y_new = next(iter(eval_loader))
X_new = X_new[:3].to(device)
with torch.no_grad():
    y_pred_logits = model(X_new)

y_pred = y_pred_logits.argmax(dim=1) # index of the largest logit
print(f"\nPrediccion con mayor confianza: {y_pred}")
print(f"Etiquetas de las predicciones: {[train_and_valid_data.classes[index] for index in y_pred]}\n")

import torch.nn.functional as F
# Si interesa las probabilidades de cada etiqueta, se pueden conseguir implementando un 
# softmax sobre las predicciones del modelo. En este caso, el modelo esta muy confiado en las 
# dos primeras predicciones
y_proba = F.softmax(y_pred_logits, dim=1)
y_proba.round(decimals=3)

# Tambien es interesante escoger unicamente las k predicciones con mayor confianza, esto se
# hace mediante la llamada a torch.topk()
y_top4_logits, y_top4_indices = torch.topk(y_pred_logits, k=4, dim=1)
y_top4_probas = F.softmax(y_top4_logits, dim=1)
y_top4_probas.round(decimals=3)
