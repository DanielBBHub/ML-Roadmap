import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
import torchmetrics
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Cargamos un modelo preentrenado en la tarea de FashionMINST con una función utilitaria
from SNL import loadWeights
model_A = loadWeights("fashion_preentrenado")

# Importamos el modulo copy con lo que extraeremos las capas del modelo A
# para utilizarlas en modelo B
import copy

torch.manual_seed(42)
# deepcopy() es una funcion para copiar todos los modulos de nn.Sequential
# menos la primera capa
reused_layers = copy.deepcopy(model_A[:-1])
model_B_on_A = nn.Sequential(
    *reused_layers,
    nn.Linear(100, 1) # new output layer for task B
).to(device)

# El siguiente paso sería entrenar el modelo, pero como los pesos de la nueva capa
# de salida se han inicializado aleatoriamente, habría que congelar las capas reusadas
# y hacer backprop para la capa de salida
# Con este for iteramos todos los submodulos del modelo, menos la capa de salida, y definimos
# requires_grad = False para no aplicar el backprop sobre ellas
for layer in model_B_on_A[:-1]:
    for param in layer.parameters():
        param.requires_grad = False

# Ya estaría lista para entrenar, pero dado que la tarea ha cambiado (clasificacion multiclase -> clasificacion binaria)
# cambiaremos le funcion de perdida y la metrica de desempeño
from torch.utils.data import Dataset, DataLoader, Subset
class BinaryTshirtPullover(Dataset):
    """
    Mantiene solo clases:
      - 0 (T-shirt/top) -> 1 (positivo)
      - 2 (Pullover)    -> 0 (negativo)
    """
    def __init__(self, base_dataset):
        self.base = base_dataset

        # targets en FashionMNIST suele estar en base_dataset.targets
        targets = torch.as_tensor(self.base.targets)

        # índices de muestras de clases 0 o 2
        mask = (targets == 0) | (targets == 2)
        self.indices = torch.where(mask)[0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.base[int(self.indices[i])]
        # binarización: 0->1 (T-shirt), 2->0 (Pullover)
        y_bin = 1 if int(y) == 0 else 0
        return x, torch.tensor(y_bin, dtype=torch.long)


# --- ejemplo de uso ---
toTensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

train_full = torchvision.datasets.FashionMNIST(root="datasets", train=True, download=True, transform=toTensor)
test_full  = torchvision.datasets.FashionMNIST(root="datasets", train=False, download=True, transform=toTensor)

train_bin = BinaryTshirtPullover(train_full)
test_bin  = BinaryTshirtPullover(test_full)

# En este caso num_workers=0 para no crear un main enorme
train_loader = DataLoader(train_bin, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
eval_loader  = DataLoader(test_bin, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

torch.manual_seed(42)
learning_rate = 0.002 

xentropy = nn.BCEWithLogitsLoss()
accuracy = torchmetrics.Accuracy(task="binary").to(device)
optimizer = torch.optim.Adam(model_B_on_A.parameters(), lr=1e-3, weight_decay=1e-4)

from ModelUtl.Train import train_minibatch_gd
train_minibatch_gd(model_B_on_A, optimizer, xentropy, train_loader, eval_loader, 30,  device, accuracy) 

model_B_on_A.eval()
X_new, y_new = next(iter(eval_loader))
X_new = X_new[:3].to(device)
with torch.no_grad():
    y_pred_logits = model_B_on_A(X_new)

y_pred = y_pred_logits.argmax(dim=1) # index of the largest logit
print(f"\nPrediccion con mayor confianza: {y_pred}")
print(f"\nEtiqueta a predecir: {y_new[:3]}")
