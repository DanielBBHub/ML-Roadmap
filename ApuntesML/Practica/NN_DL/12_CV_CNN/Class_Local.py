# Classification and Localization
# La localización de objetos se puede expresar en problemas de regresion lineal; predecir el centro de la bounding box o 
# incluso su altura y anchura o las coordenadas que la delimiten (esquinas superior izquierda e inferior derecha). En definitiva
# esto quiere decir que necesitamos una capa adicional con una salida de cuatro numeros

import torch
import torch.nn as nn

class FlowerLocator(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.localization_head = nn.Sequential(
        nn.Flatten(),
        nn.Linear(base_model.classifier[2].in_features, 4)
        )

    # El metodo forward recibe un lote de imagenes preprocesadas y genera una predicción para cada una de las clases asi
    # como las coordenadas de la caja que contienen estas a estas clases
    def forward(self, X):
        features = self.base_model.features(X)
        pool = self.base_model.avgpool(features)
        logits = self.base_model.classifier(pool)
        bbox = self.localization_head(pool)
        return logits, bbox
        
# Para el entrenamiento de la segunda cabeza habria que tener etiquetadas las cajas que contengan a estas flores, ya bien sea a mano
# o con herramientas que ayuden a acelerar el proceso. Una vez etiquetadas habria que crear un nuevo conjunto de datos, donde cada
# registro tuviese una imagen, etiqueta y bounding box. TorchVision ya tiene una clase para BoundingBoxes que representa una
# lista de ellas.

import torchvision.tv_tensors
bbox = torchvision.tv_tensors.BoundingBoxes(
    [[377, 199, 248, 262]], # center x=377, center y=199, width=248, height=262
    format="CXCYWH", # other possible formats: "XYXY" and "XYWH"
    canvas_size=(500, 754) # raw image size before preprocessing
    )

