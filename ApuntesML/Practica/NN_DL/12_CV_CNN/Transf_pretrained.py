# Using pretrained moedls for transfer learning
import torchvision
from functools import partial
import torch
import torch.nn as nn
import torchmetrics
pesos = torchvision.models.ConvNeXt_Base_Weights.IMAGENET1K_V1
model = torchvision.models.convnext_base(weights = pesos).to("cuda")

# Este conjunto unicamente tiene 10 imagenes por clase, con lo que la mejor opcion sera utilizar un modelo
# ya entrenado para aplicar transfer learning y reentrenar con las imagenes nuevas
DefaultFlowers102 = partial(torchvision.datasets.Flowers102, root="datasets",
    transform=pesos.transforms(), download=True)

valid_set = DefaultFlowers102(split="val")
test_set = DefaultFlowers102(split="test")

from torch.utils.data import DataLoader

valid_loader = DataLoader(valid_set, batch_size=32)
test_loader = DataLoader(test_set, batch_size=32)

from flowerclasses import flowers102_classes

# Para adaptar el modelo ConvNeXt-base al nuevo dataset habra que modificar la estructura preparada para
# predecir 1000 clases a 102; "cortarle la cabeza" al modelo y reemplazarla con las capas necesarias
print(f"Submodulos del modelo: {[name for name, child in model.named_children()]}")

# El modulo de features es la parte central del modelo, el que incluye todas las capas menos 
# las avg pool y "la cabeza" del modelo
# Con model.classifier podemos ver que es un modulo nn.Sequential compuesto por una capa layer norm, Flatten y Linear con 1024 entradas
# y 1000 salidas. 
print(f"Detalles modelo: {model.classifier}")

# Esta ultima es la capa de salida que tendremos que reemplazar modificando el numero de salidas, en este caso 102 por el dataset
model.classifier[2] = nn.Linear(1024, 102).to("cuda")

# El siguiente paso sera congelar las capas preentrenadas, por lo menos durante el inicio del entrenamiento. Haremos esto
# congelando todos los parametros del modelo
for param in model.parameters():
    param.requires_grad = False

# Para luego descongelar la "cabeza" (layer norm, flatten y la capa de salida)
for param in model.classifier.parameters():
    param.requires_grad = True


# Para conseguir algo mas de precision siempre es una buena idea aplicar data augmentation en las imagenes de entrenamiento, como 
# rotaciones, recortes ...
import torchvision.transforms.v2 as T
transforms = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=30),
    T.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_set = DefaultFlowers102(split="train", transform=transforms)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
from ModelUtl.Train import train_minibatch_gd

xentropy = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=102).to("cuda")
train_minibatch_gd(model, optimizer, xentropy, train_loader, valid_loader, 15, "cuda", accuracy)

# Otras ideas para seguir mejorando la precision del modelo serian:
#   - Probar otros modelos preentrenados 
#   - Buscar y etiquetar nueva informacion
#   - Crear un ensablado de modelos y combinar sus predicciones
#   - Analizar casos erroneos, sus caracteristicas e intentar modificar el preprocesado para 
#   atacar estas debilidades
#   - Utilizar un performance scheduling
#   - Descongelar las capas gradualmente, de arriba a abajo
#   - Utilizar ratios de aprendizaje diferenciales; mas pequeños para las capas mas bajas y subirlo a medida que subes en las capas
#   - Utilizar otro optimizador
#   - Utilizar otras tecnicas de regularizacion