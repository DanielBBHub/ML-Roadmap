# SAVING AND LOADING PYTORCH MODELS

# La forma mas sencilla de guardar un modelo entrenado con pytorch es utilizar el metodo pytorch.save(). El resultado de esta llamada
# sera un archivo comprimido en disco con el modelo y sus pesos. La convencion es poner las extensiones como
#   -.pt
#   -.pth
import torch

# .save() convierte el objeto en una serie de bytes mediante el modulo pickle
def saveModel(model, name):
    torch.save(model,name + ".pt")

# El argumento de weights_only=False se asegura que se carga el modelo por completo,
# con el que puedes hacer inferencia
def loadModel(name):
    return torch.load(name + ".pt", weights_only=False)

# Hay dos problemas con hacer la carga/guardado de esta manera:
#     - La serializacion de pickle es insegura, de forma en que se puede insertar codigo. Siempre que se cargue un modelo de esta manera
#     hay que estar seguros de confiar en la fuente
#     - La libreria es inestable entre versiones y puede romperse si el entorno de carga tiene una estructura diferentes

# Para evitar estos problemas se recomienda guardar y cargar solo los pesos del modelo

# El "state dict" es un diccionario ordenado que contiene todos los parametros que puedan obtenerse con named_parameters(),
# asi como buffers si es que el modelo lo tiene. (Un buffer es un tensor registrado con register_buffer() en un modelo). 
# Estos contieneninformacio adicional
def saveWeights(model, name):
    model_data = {
    "model_state_dict": model.state_dict(),
    "model_hyperparameters": {"n_inputs": 1 * 28 * 28, "n_hidden1": 300, [...]}
    }
    torch.save(model_data, name + ".pt")

# Para cargar los pesos, hay que crear un modelo con la misma estructura, cargar el diccionario de estado y
# añadirlo con .load_state_dict() en el modelo copia
def loadWeights(name):
    loaded_data = torch.load("my_fashion_mnist_model.pt", weights_only=True)
    new_model = ImageClassifier(**loaded_data["model_hyperparameters"])
    new_model.load_state_dict(loaded_data["model_state_dict"])
    return new_model