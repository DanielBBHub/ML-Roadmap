# COMPILING AND OPTIMIZING A PYTORCH MODEL

# Pytorch tiene la caracteristica de convertir el codigo del modelo a torchscript, lo cual tiene beneficios:
    # - El codigo en TorchScript puede ser compilado y optimizado para producir modelos mas grandes 
    # - TorchScript puede ser serializado y cargado para ejecutar el codigo en entornos de python
    #   y C++.

# Hay dos maneras de convertir un modelo a TorchScript:
    # - Tracing: Pytorch corre el modelo con unas muestras, escribe cada operacion y convierte el log en TorchScript.
    #   Esto funciona bien en modelos estaticos, es decir, en aquellos en el que su forward() no tiene 
    #   condiciones o bucles.

def tracing(model, x_new):
    torchscript_model = torch.jit.trace(model, x_new)

    # - Scripting: Pytorch convierte el codigo de python a TorchScript. Este metodo es el correcto para modelos con 
    #   condiciones y bucles mientras estos sean tensores. La pega de esta conversion es que no soporta estructuras de datos
    #   complejas (generators, list comprehension ...)

def scripting(model):
    torchscript_model = torch.jit.script(model)

# Es importante resaltar que los modelos en TorchScript unicamente se pueden utilizar para inferencia, no tienen soporte
# en el entorno como para llevar cuenta de los gradientes o los parametros, pero se puede optimizar para la inferencia
# mediante torch.jit.optimize_for_inference(torchscript_model)

# Por otro lado, TorchScript ya ha acabado su desarrollo, aun siendo la mejor opcion para correr modelos en SW embedido,
# PyTorch ha desarrollado un conjunto de herramientas centradas en torch.compile(model). El resultado de esta compilacion
# es un modelo compilado Just-In-Time (JIT), es decir, que se compilara y optimizara cuando lo uses. Entre bambalinas,
# torch.compile() utiliza la informacion del grafo de computacion, obtenida con TorchDynamo, en tiempo real para
# manejar condicionales y bucles para optimizar mejor el modelo.