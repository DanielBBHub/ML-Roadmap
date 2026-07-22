""" Exercises
1. PyTorch is similar to NumPy is many ways, but it offers some extra features. Can
you name the most important ones?
- Autodiferenciación con la implementación del modulo autograd
- Aceleración por HW
- Implementaciones de optimizadores y componentes de redes neuronales

2. What is the difference between torch.exp() and torch.exp_(), or between2.
torch.relu() and torch.relu_()?
torch.exp() devuelve una copia del tensor de entrada, mientras que exp_() lo modifica, lo mismo
para relu() y relu_()

3. What are two ways to create a new tensor on the GPU?
# Definicion de un tensor y copia de este a la GPU [to(device)]
M = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
M = M.to(device)

# Definicion directa de un tensor en grafica
M2 = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device=device)

4. What are three ways to perform tensor computations without using autograd?
- Manipular los tensores creados con requires_grad=False
- Correr los calculos dentro de un bucle torch.no_grad(): .
- Llamar a la funcion detach() del tensor a manipular con autograd.

5. Will the following code cause a RuntimeError? What if you replace the second 
line with z = t.cos_().exp()? And what if you replace it with z =
t.exp().cos_()?
t = torch.tensor(2.0, requires_grad=True)
z = t.cos().exp_()
z.backward()
En el primer ejemplo no habria ningun error, ya que en la linea 2 "t.cos()" devuelve una copia del nodo
sobre el cual se ejecuta "exp_()", mientras que si lo modificas por "t.cos_().exp()", en este caso estaras 
trabajando sobre el nodo creado, uno que requiere autograd. Por ultimo, si lo cambias por "t.exp().cos_()",
volvera a lanzar RuntimeError dado que con exp() no has guardado el tensor e intentas, otra vez, modificar
un nodo con autograd en "el sitio"

How about the following code, will it cause an error? And what if you replace the
third line with w = v.cos_() * v.sin()? Will w have the same value in both
cases?
u = torch.tensor(2.0, requires_grad=True)
v = u + 1
w = v.cos() * v.sin_()
w.backward()
Este ejemplo si levantara RuntimeError ya que al ejevutar v.cos() reservas el tensor
para la operación y cuando vas a ejecutar sin_() el programa para por que vas a modificar
v sabiendo que la necesitas para cos(_v_)
Por otro lado, al modificar el codigo con "v.cos_() * v.sin()" y alteras el orden, eliminas
la incompatibilidad del registro del nodo y ejecuta.
Por ultimo, como v.cos_() modifica directamente el registro, no dará el mismo valor a w, en una operacion
tendra cos(3) * sin(3) y en otra cos(3) * sin(cos(3))

6. Suppose you create a Linear(100, 200) module. How many neurons does it
have? What is the shape of is weight and bias parameters? What input shape
does it expect? What output shape does it produce?
Tiene 200 neuronas, una por cada salida. El tensor de pesos tiene una forma [200,100] y el de 
sesgos tiene una forma [200]. Espera una entrada de [..., 100] y devuelve una salida de [..., 200]
La forma de la salida y la entrada son iguales, pero escaladas por 2, si la entrada tiene una forma
de [1,32,100], la salida sera [1,32,200]
7. What are the main steps of a PyTorch training loop?
- Preparar los conjuntos de informacion (Train, test, val) 
- Definir los Dataloaders para los sets
- model.train()
- Enviar los batches a GPU (_batch = X_batch.to(device, non_blocking=True))
- Calcular prediccion
- Calcular perdida de prediccion
- Gradiente + optimizer.step(), optimizer.zero_grad()
- Adicionalmente puede evaluarse el modelo con model.eval() y un bucle con torch.no_grad()

8. Why is it recommended to create the optimizer after the model is moved to the
GPU?
Es recomendable crearlo despues de mover el modelo a la GPU por que el estado que tiene el optimizador
se crea en el dispsitivo donde estan los parametros del modelo

9.What DataLoader options should you generally set to speed up training when
using a GPU?
train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        pin_memory=(device == "cuda"),      # pin_memory útil sobre todo en CUDA
        num_workers=num_workers,             # workers para carga en paralelo de lotes 
        prefetch_factor=2 if num_workers > 0 else None,  # prefech para limitar el tiempo entre lote y lote
        persistent_workers=(num_workers > 0) # evita recrear workers cada epoch
    )

10. What are the main classification losses provided by PyTorch, and when should
you use each of them?


11. Why is it important to call model.train() before training and model.eval()
before evaluation?


12. What is the difference between torch.jit.trace() and torch.jit.script()?12.


13. Use autograd to find the gradient vector of f(x, y) = sin(x2 y) at the point (x, y) =13.
(1.2, 3.4).


14. Create a custom Dense module that replicates the functionality of an nn.Linear14.
module followed by an nn.ReLU module. Try implementing it first using the
nn.Linear and nn.ReLU modules, and then reimplement it using nn.Parameter
and the relu() function.


15. Build and train a classification MLP on the CoverType dataset:15.
a. Load the dataset using sklearn.datasets.fetch_covtype() and create aa.
custom PyTorch Dataset for this data.

b. Create data loaders for training, validation, and testing.b.

c. Build a custom MLP module to tackle this classification task. You can option‐c.
ally use the custom Dense module from the previous exercise.

d. Train this model on the GPU, and try to reach 93% accuracy on the test set.d.
For this, you will likely have to perform hyperparameter search to find the
right number of layers and neurons per layer, a good learning rate and batch
size, and so on. You can optionally use Optuna for this. """