---
tipo: concepto
categoria: deep-learning
subcategoria: arquitecturas
tags:
  - cnn
  - convolucion
  - pooling
  - softmax
  - dropout
  - autocoding
seccion_roadmap: Deep Learning Architectures
orden_seccion: 13
orden: 1
dificultad: avanzada
---
## Redes neuronales profundas

### Capas de entrada

Las **deep nets** o redes neuronales convolucionales, en el caso de estar procesando una imagen, recorren las filas de valores de la muestra inicial utilizando un **kernel** o ventana de búsqueda (por ejemplo, 10x10 px) y asignan un valor a estas regiones antes de desplazar la ventana al siguiente valor, con cierto solapamiento respecto a la región anterior.

Una vez se han obtenido todos los valores de todas las regiones, se vuelve a iterar sobre estos resultados con la misma u otra ventana para quedarse con un valor concreto —puede ser el máximo, el mínimo, etc.—, obteniendo así una lista reducida de valores finales. Este proceso puede repetirse múltiples veces.

Finalmente, estos resultados se introducen en una red neuronal que calculará:

F(clase)

es decir, la probabilidad de que esta entrada pertenezca a una de las clases preentrenadas.

A los diferentes pasos de este proceso se les conoce como:

1. **Convolución**
2. **Pooling**
3. **Predicción**

![[Pasted image 20260330163531.png]]

---

### Capas intermedias

[](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/blob/main/Apuntes%20ML/README.md#capas-intermedias)

El concepto de **autocoding** es interesante porque, cuando igualas los valores deseados (d) a los valores de entrada (x) y entrenas la red para que la entrada sea igual a la salida, pero con una capa intermedia más pequeña, significa que dicha capa intermedia está siendo capaz de **codificar una generalización de los parámetros de entrada**.

Es decir, con menos neuronas se consigue reproducir un resultado fiable, lo que implica un alto porcentaje de acierto.

![[Pasted image 20260330163453.png]]

---

### Capas de salida

[](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/blob/main/Apuntes%20ML/README.md#capas-de-salida)

En cuanto al resultado de la predicción, este tiene en cuenta un umbral (T) y una entrada (X) multiplicada por un peso (W), todo lo cual entra en un sumatorio y pasa por una función sigmoide que decantará el cálculo de esa neurona.

Dado que la función sigmoide de desempeño depende del peso y del umbral ((W) y (T)), el cambio en estos valores puede desplazar la función o hacer que su pendiente sea mucho más empinada.

![[Pasted image 20260330164810.png]]

Se puede pensar en los diferentes puntos de la función sigmoide como **la probabilidad de encontrar una muestra en la información**. Puesto que esta función depende de pesos y umbrales, el objetivo es ajustar estos valores para aumentar la probabilidad de encontrar las clases a predecir en el resultado de la neurona.

Finalmente, la salida de las (n) neuronas que conforman la capa de salida será (F(C_n)), siendo esto la probabilidad de cada una de las clases. El resultado será, por tanto, un vector que asigna las probabilidades de la muestra de entrada a cada una de las clases preentrenadas.

Una vez tengamos este vector, podremos normalizar estas probabilidades de la siguiente manera:

P(C1)=F(C1)∑F(Ct)

Siendo la probabilidad de una clase el resultado de la función para esa clase, normalizado por el sumatorio de los resultados de las funciones del resto de clases. Esto se conoce como **Softmax**.

---

### Dropout

[](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/blob/main/Apuntes%20ML/README.md#dropout)

Más tarde, en el I+D de redes neuronales, se descubrió el concepto de **Dropout**, por el cual en cada iteración se “desactivan” un conjunto de neuronas. Con ello se puede evitar llegar a un máximo local durante la fase de entrenamiento de la red neuronal.

---

### Robustez frente a perturbaciones

[](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/blob/main/Apuntes%20ML/README.md#robustez-frente-a-perturbaciones)

Para acabar, se hicieron varias comprobaciones cambiando sutilmente píxeles en ciertas imágenes en las que había una alta confianza en la clase a la que pertenecía el objeto, cambiando por completo el resultado.

![[Pasted image 20260330172115.png]]

De igual manera, se deformaron completamente otras imágenes, hasta el punto de ser irreconocibles, pero la red neuronal las predecía como la clase en concreto.

![[Pasted image 20260330172313.png]]

Lo cual indica que la red neuronal es capaz de **“desenterrar” las pruebas locales suficientes** como para poder predecir una clase aunque esta no se encuentre realmente en la imagen.
