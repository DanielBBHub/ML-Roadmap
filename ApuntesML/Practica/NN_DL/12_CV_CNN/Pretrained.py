# Using TorchVision's pretrained models
# No hace falta implementar las arquitecturas mas populares ya que Torchvision tiene implementaciones de estas
# para utilizarlas con apenas un par de lineas de codigo
import torchvision
import torch
import numpy as np
from sklearn.datasets import load_sample_images

sample_images = np.stack(load_sample_images()["images"])
sample_images = torch.tensor(sample_images, dtype=torch.float32) / 255
sample_images_permuted = sample_images.permute(0, 3, 1, 2)

# Estas lineas de codigo descarga automaticamente los pesos desde "Torch Hub" y estos quedan cacheados
# para utilizarlos en un futuro. Hay modelos con pesos reentrenados que tienen varias versiones (p.j IMAGENET1K_V2)
pesos = torchvision.models.ConvNeXt_Base_Weights.IMAGENET1K_V1
model = torchvision.models.convnext_base(weights = pesos).to("cuda")

# Para la lista entera de modelos preentrenados se puede ejecutar la siguiente linea
# print(f"Modelos preentrenados: {models.list_models}")

# Por otro lado, para conseguir todos los pesos disponibles para un modelo en concreto, como convnext_base
# puede ejectuarse la siguiente linea
# print(f"Pesos de {modelo}: {list(models.get_model_weights(modelo))}")

# Se recomienda utilizar las transformaciones que vengan con el modelo para que las entradas a este sean
# con las que se ha entrenado. Esto se puede hacer facilmente con estas lineas
transf = pesos.transforms()
pre_img = transf(sample_images_permuted)

# Finalmente, podemos poner el modelo en modo evaluación y ejecutar la inferencia sobre las imagenes 
model.eval()
with torch.no_grad():
    y_logits = model(pre_img.to("cuda"))

# El resultado de la inferencia es un tensor 2 x 1000 con los logits para cada una de las imagenes
y_pred = torch.argmax(y_logits, dim=1)
print(f"Predicciones: {y_pred}")

class_names = pesos.meta["categories"]
print(f"Clases de las predicciones: {[class_names[class_id] for class_id in y_pred]}")