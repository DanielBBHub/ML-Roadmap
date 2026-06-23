import torch
import numpy as np

# La estructura de datos basica de Pytorch son los tensores (arrays multidimensionales). Estos tensores pueden tener cualquier valor numérico
# y una vez inicializado se escogerá el tipo de variable dependiendo de cual es mas generica (complejo > float > integer > bool)
X = torch.tensor([[1.0, 4.0, 7.0], [2.0, 3.0, 6.0]])
print(f"\nTensor de Pytorch: {X}")

print(f"\nForma del tensor [{X.shape}] y su tipo [{X.dtype}]")

# La indexacion de los tensores funciona de la misma manera que en numpy
print(f"\nValor en la posicion [0,1]: {X[0, 1]}")
print(f"\nTodos los valores en la posición 1: {X[:, 1]}")

# Tambien es capaz de realizar calculos sobre los tensores
# Sumar 1 a todos los valores del tensor y multiplicarlos por 10
prod_sum = 10 * (X + 1.0)
print(f"\nSuma 1 y escala por 10 los valores del tensor: {prod_sum}")

# Funcion exp
exp = X.exp()
print(f"\nExponente a los valores del tensor: {exp}")

# Multiplicacion de matrices y matriz transpuesta
I = X @ X.T
print(f"\nResultado de la multiplicacion de matrices X y X^t: {I}")

# Tambien se pueden convertir tensores a arrays de numpy y viceversa
print(f"\nConversion de tensor a array de numpy: \n{X} \n-----------> \n{X.numpy()}")
print(f"\n Conversión de array de numpy a tensor: \n{np.array([[1., 4., 7.], [2., 3., 6.]])} \n-----------> \n{torch.tensor(np.array([[1., 4., 7.], [2., 3., 6.]]))}")
# La precision de los floats en Pytorch comparada con la de numpy es de 32 a 64. Lo recomendable es especificar dtype=torch.float32 
# o utilizar torch.FloatTensor() para convertir estos arrays ya que ocupa la mitad de espacio y la red no necesitara este tipo de precision

# Pytorch provee muchas operaciones sobre los tensores, como abs_(), sqrt_() y zero_() o incluso relu_() para modificar el tensor de entrada
X[:, 1] = -99
print(f"\nModificacion de todos los valores en posicion 1 por -99: {X}")
X.relu_()
print(f"\nModificacion con funcion relu_() de todos los valores negativos: {X}")