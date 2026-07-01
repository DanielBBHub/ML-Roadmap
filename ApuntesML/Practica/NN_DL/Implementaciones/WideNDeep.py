## BUILDING NONSECUENTIAL  MODELS USING CUSTOM MODULES
import torch.nn as nn
import torch
# Las redes neuronales "anchas y profundas" conectan todas/algunas salidas directamente con la capa de salida,
# lo cual permite que se reconozcan los patrones mas básicos, mediante los caminos cortos, asi como los patrones
# mas profundos, mediante la interconexion de las capas intermedias

# Para implementar modulos personalizados, podemos utilizar el modulo "Module" de PyTorch como superclase
# e implementar todas las capas necesarias en el constructor y definir como estas capas deben ser utilizadas en forward()

# "Module" tiene los siguientes metodos:
#   - children()
#   - named_children()
# los cuales se utilizan para recibir un iterador de python, el cual sera el submodulo ModuleList o ModuleDict

# Por otro lado, tambien tiene los metdos:
#   - parameters()
#   - named_parameters()
# que se utilizan para recibir un iterador donde se guardan el numero variable de parametros que tenga el modulo, 
# guardados dentro de los modulos ParameterList o ParameterDict

class WideAndDeep(nn.Module):
    # Constructor donde se definen las capas
    def __init__(self, n_features):
        super().__init__()
        # Propiedad de "Module" donde defines las capas de la red,
        # en este caso una red secuencial con una capa de entrada y una intermedia
        # Esta sera la parte "profunda" de nuestro modelo
        self.deep_stack = nn.Sequential(
        nn.Linear(n_features, 50), nn.ReLU(),
        nn.Linear(50, 40), nn.ReLU(),
        )
        # Propiedad de "Module" que define la capa de salida, la entrada es
        # 40 + nº caracteristicas para poder concatenar la salida de la red y la entrada del modelo 
        self.output_layer = nn.Linear(40 + n_features, 1)

    def forward(self, X):
        # Se le mete como entrada X a la red "profunda"
        deep_output = self.deep_stack(X)
        # Concatenas la entrada X y la salida de la red "profunda" 
        # para usarla como entrada de la capa de salida
        wide_and_deep = torch.concat([X, deep_output], dim=1)
        return self.output_layer(wide_and_deep) 


class WideAndDeepV2(nn.Module):
# Constructor donde se definen las capas
    def __init__(self, n_features):
        super().__init__()
        # Propiedad de "Module" donde defines las capas de la red,
        # en este caso una red secuencial con una capa de entrada y una intermedia
        # Esta sera la parte "profunda" de nuestro modelo
        self.deep_stack = nn.Sequential(
        nn.Linear(n_features - 2, 50), nn.ReLU(),
        nn.Linear(50, 40), nn.ReLU(),
        )
        # Propiedad de "Module" que define la capa de salida, la entrada es
        self.output_layer = nn.Linear(40 + 5, 1)
    
    def forward(self, X):
        X_wide = X[:, :5]
        X_deep = X[:, 2:]
        deep_output = self.deep_stack(X_deep)
        wide_and_deep = torch.concat([X_wide, deep_output], dim=1)
        return self.output_layer(wide_and_deep)

### BUILDING MODELS WITH MULTIPLE INPUTS

# En algunos casos es necesario tener diferentes tipos de inputs, ya que no pueden definirse en un mismo tensor,
# por ejemplo, cuando las posibles entradas son texto e imagenes. Para adaptar una red a este requerimiento podemos
# modificar el metodo forward() de la clase

class WideAndDeepV3(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        # Propiedad de "Module" donde defines las capas de la red,
        # en este caso una red secuencial con una capa de entrada y una intermedia
        # Esta sera la parte "profunda" de nuestro modelo
        self.deep_stack = nn.Sequential(
        nn.Linear(n_features - 2, 50), nn.ReLU(),
        nn.Linear(50, 40), nn.ReLU(),
        )
        # Propiedad de "Module" que define la capa de salida, la entrada es
        self.output_layer = nn.Linear(40 + 5, 1)

    def forward(self, X_wide, X_deep):
        deep_output = self.deep_stack(X_deep)
        wide_and_deep = torch.concat([X_wide, deep_output], dim=1)
        return self.output_layer(wide_and_deep)

# Cuando el modelo tiene varios inputs es facil no acordarse del orden y cometer un error. Para evitarlo
# podemos definir un Dataset personalizado que devuelva un diccionario de nombres de inputs - valores
class WideAndDeepDataset(torch.utils.data.Dataset):
    def __init__(self, X_wide, X_deep, y):
        self.X_wide = X_wide
        self.X_deep = X_deep
        self.y = y

    def __len__(self):
        return len(self.y)
    # Modificacion de la clase base Dataset para definir la devolucion de 
    # los inputs "anchos" y "profundos" por nombre-valor
    def __getitem__(self, idx):
        input_dict = {"X_wide": self.X_wide[idx], "X_deep": self.X_deep[idx]}
        return input_dict, self.y[idx]