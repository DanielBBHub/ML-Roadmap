# Es una implementacion sencilla, pero es mejor definir una envoltura a tu nn.Secuential que definir el modelo directamente asi,
# con el objetivo de facilitar el mantenimiento y configuracion
import torch.nn as nn

class ImageClassifier(nn.Module):
    def __init__(self, n_inputs, n_hidden1, n_hidden2, n_classes):
        super().__init__()
        self.mlp = nn.Sequential(
        # Flatten para comprimir todas las dimensiones del tensor en 1
        nn.Flatten(),
        # El resto lineales con funciones de activacion ReLU
        nn.Linear(n_inputs, n_hidden1),
        nn.ReLU(),
        nn.Linear(n_hidden1, n_hidden2),
        nn.ReLU(),
        nn.Linear(n_hidden2, n_classes)
        )

    def forward(self, X):
        return self.mlp(X)