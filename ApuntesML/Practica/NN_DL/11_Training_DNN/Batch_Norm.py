import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
import torchmetrics
from ModelUtl.Train import train_minibatch_gd

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

toTensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

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

# Para utilizar las capas de normalización implementadas en PyTorch simplemente hay que añadir 
# nn.BatchNorm1d antes o despues de cada función de activación de las capas intermedias.
# También es posible añadir una capa BN como primera capa, eliminando la necesidad de estandarizar
# las entradas manualmente
# En el paper los investigadores abogan por una estructura con las capas BN antes de las funciones de activación
# Si lo hacemos de esta manera, podemos definir el parametro bias=False, ya que las capas BN ya tienen uno propio
model = nn.Sequential(
nn.Flatten(),
nn.Linear(1 * 28 * 28, 300, bias=False),
nn.BatchNorm1d(300),
nn.ReLU(),
nn.Linear(300, 100, bias=False),
nn.BatchNorm1d(100),
nn.ReLU(),
nn.Linear(100, 10)
)

model = model.to(device)
xentropy = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
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

# Si miramos a los parametros de las capas BN, veremos que tiene "weight" y "bias", 
# los cuales corresponden a "γ" y "β"
print(dict(model[1].named_parameters()).keys())

# Si miramos a los buffers de estas capas BN, veremos tres running_mean, running_var, and num_batches_tracked,
# las dos primeras son "μ" and "σ²"
print(dict(model[1].named_buffers()).keys())