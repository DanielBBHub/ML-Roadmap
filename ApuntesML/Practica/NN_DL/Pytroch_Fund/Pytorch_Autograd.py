import torch
# Pytorch tiene una implementacion para la auto diferenciacion llamada autograd, 
# con la cual permite calcular los gradientes
# Definiendo el tensor con "requires_grad=True" se especifica que este sera una variable
x = torch.tensor(5.0, requires_grad=True)
# Definicion de f como el tensor x al cuadrado. Aun que en este estado el valor de f sea 25
# la variable contiene una propiedad grad_fn que guarda la operacion del tensor (potencia en este caso)
f = x ** 2
print(f"\nResultado de calcular f(x) para x=5: {f}")
# Cálculo del gradiente de f
f.backward()
print(f"\nResultado de calcular f'(x) para x=5: {x.grad}")

# Despues de calcular el gradiente se puede calcular el paso de descenso del gradiente
# restando una parte del gradiente a las variables del modelo

learning_rate = 0.1
# Con .no_grad() se desactiva el seguimiento del gradiente para no afectarlo,
# unicamente al valor de la variable. En este caso la variable bajaria 
# 0.1 * 10.0 = 1.0; 5 ---> 4
with torch.no_grad():
    x -= learning_rate * x.grad # gradient descent step z
print(f"\nResultado de calcular el descenso con no_grad(): {x}")
# Tambien se puede conseguir el mismo resultado con .detach(), la cual crearia
# una copia del tensor original, con el mismo valor y apuntando al mismo espacio de 
# memoria, con lo que si modificas este valor, modificarías el tensor original
x_detached = x.detach()
x_detached -= learning_rate * x.grad
print(f"\nResultado de calcular el descenso con detach()[x_detached]: {x_detached}")
print(f"\nResultado de calcular el descenso con detach()[x]: {x}")

# Utilizar detach() puede venir bien cuando necesitas hacer calculos en el tensor sin afectar al gradiente
# o cuando necesitas controlar manualmente cuales operaciones deberian contribuir al gradiente. 
# Por otro lado, no_grad() se utiliza cuando realizas inferencia o el descenso de gradiente.

# Para acabar, y antes de volver a repetir el proceso, es importante poner a 0 los gradientes de los parametros del 
# modelo, en caso contrario, se acumulara error en cada iteracion del entrenamiento
x.grad.zero_()

# El bloque de entrenamiento luce algo como lo siguiente
""" 
learning_rate = 0.1
x = torch.tensor(5.0, requires_grad=True)
for iteration in range(100):
    f = x ** 2 # forward pass
    f.backward() # backward pass
    with torch.no_grad():
        x -= learning_rate * x.grad # gradient descent step
    
    x.grad.zero_() # reset the gradients 
"""

# Operaciones como:
    # - exp()
    # - relu()
    # - rsqrt()
    # - sigmoid()
    # - sqrt()
    # - tan()
    # - tanh()
# guardan sus salidas en el grafo de calculos en el paso hacia delante y las utilizan para calcular el gradiente 
# durante el paso hacia atras, es decir, no se han de modificar los valores en el sitio o saltara RuntimeError

    # - abs()
    # - cos()
    # - log()
    # - sin()
    # - square()
    # - var()

# guardan las entradas, en vez de las salidas, con lo que las operaciones "en el sitio" no impactan ya que modifican la salida, no su valor

    # - max()
    # - min()
    # - norm()
    # - prod()
    # - sgn()
    # - std()

# guardan las entradas y salidas, con lo que no se han de modificar ninguna de las dos

    # - ceil()
    # - floor()
    # - mean()
    # - round()
    # - sum()

# no guardan ni la entrada ni la salida, se puede modificar cualquiera