""" Exercises
1. What is the problem that Glorot initialization and He initialization aim to fix?
El problema que se pretende solucionar es la explosión/desvanecimiento de los gradientes en la 
fase de entrenamiento intentando mantener la desviación estandar de la salida lo mas
parecida a la de la entrada posible

2. Is it OK to initialize all the weights to the same value as long as that value is
selected randomly using He initialization?
No. La red neuronal tendría símetria en sus capas y estas quedarían reducidas 
a una neurona pracitcamentey acabaría resultando en un modelo subóptimo al final del entrenamiento

3. Is it OK to initialize the bias terms to 0?
Si, se puede inicializar los valores de sesgo a 0, no tiene ningún impacto en el entrenamiento

4. In which cases would you want to use each of the activation functions we
discussed in this chapter?
Sigmoide: en modelos de predicción binaria
ReLU: Es una buena función de activación por defecto
Variantes de ReLU: en un modelo que esta presentando neuronas muertas en el entrenamiento
o que necesita una mejora en la precisión
GLU, Swish y Mish: modelos grandes y problemas complejos
Tanh: modelo que necesite una salida fija en un rango (p.j: -1,1)

5. What may happen if you set the momentum hyperparameter too close to 1 (e.g.,
0.99999) when using an SGD optimizer?
El efecto del momentum sera irrisorio y podría presentar los mismo problemas que si no 
se aplicase la optimización por momentum; podría quedarse atascado en una meseta en el
descenso de gradiente y tardar más en acabar el entrenamiento

6. Name three ways you can produce a sparse model.
    - Regularización l1
    - Poda de neuronas
    - Quantización (reduciendo la precisión pueden quedar a 0 los pesos de las neuronas)

7. Does dropout slow down training? Does it slow down inference (i.e., making
predictions on new instances)? What about MC dropout?
Si, la técnica de dropout afecta a la velocidad de entrenamiento, ya que al desactivar neuronas en cada 
iteración, estás perdiendo conexiones posiblemente útiles y capacidad de la red en general, normalmente
por un factor de 2, pero no tiene ningún efecto en la velocidad de inferencia del modelo.

Por otro lado, el dropout Monte Carlo afecta a ambas, tanto al entrenamiento como a la inferencia,
esta última debido a que se debe hacer inferencia múltiples veces sobre la muestra para calcular
una media que será el resultado de la predicción


8. Practice training a deep neural network on the CIFAR10 image dataset:
    a. Load CIFAR10 just like you loaded the FashionMNIST dataset in Chapter 10,
    but using torchvision.datasets.CIFAR10 instead of FashionMNIST. The
    dataset is composed of 60,000 32 × 32–pixel color images (50,000 for training,
    10,000 for testing) with 10 classes.

    b. Build a DNN with 20 hidden layers of 100 neurons each (that’s too many, but
    it’s the point of this exercise). Use He initialization and the Swish activation
    function (using nn.SiLU). Since this is a classification task, you will need an
    output layer with one neuron per class.

    c. Using NAdam optimization and early stopping, train the network on the
    CIFAR10 dataset. Remember to search for the right learning rate each time
    you change the model’s architecture or hyperparameters.

    d. Now try adding batch-norm and compare the learning curves: is it converging
    faster than before? Does it produce a better model? How does it affect training
    speed?

    e. Try replacing batch-norm with SELU, and make the necessary adjustments to
    ensure the network self-normalizes (i.e., standardize the input features, use
    LeCun normal initialization, make sure the DNN contains only a sequence of
    dense layers, etc.).

    f. Try regularizing the model with alpha dropout. Then, without retraining your
    model, see if you can achieve better accuracy using MC dropout.
    
    g. Retrain your model using 1cycle scheduling and see if it improves training
    speed and model accuracy.
    """