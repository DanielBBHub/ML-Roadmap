# Perceptrón multicapa

Los perceptrones multicapa son redes neuronales que imitan las neuronas biológicas con el objetivo de encontrar patrones en la información de entrada y aprender de ella. Estos perceptrones están compuestos por entradas de información, pesos (importancias) y las funciones de activación y la salida.

Inicialmente sólo eran capaces de activarse con y realizar operaciones lógicas sencillas, como AND, OR y NOT. Con la invención de la autodiferenciación y el uso de gradientes de descenso se logró potenciar inmensamente esta tecnología, con la que poder entrenar redes neuronales que pudiesen ser entrenadas en grandes volumenes de dátos.

## Backprop

La retro-propagación es una técnica de ML que se utiliza para el aprendizaje del modelo. Mediante ella se evalua el desempeño de la red respecto a los parámetros utilizados en el entrenamiento, para modificar la importancia de cada una de las neuronas en cuanto a obtener el resultado deseado.

Esencialmente, consiste en calcular las derivadas parciales de las variables unas respecto a las otras utilizando la regla de la cadena, con lo que se modificarán los pesos.

## Funciones de activación

Las funciones de activación son aquellas funciones matemáticas que se calculan en base a los cálculos de la neurona y evaluan si esta disparará y comunicará la información o no. Hay varias:

- Escalón: Una función de activación binaria, si pasa un umbral la neurona dispar

- Sigmoide: Función de activación en forma de S $\alpha (z) = \frac{1}{(1 + exp(-z))}$ que abarca valores de $[0,1]$

- Tanh: Función de activación parecido a la sigmoide $tanh(z) = 2\sigma (2z) -1$ que centra los valores de la función entre $[-1,1]$, lo que ayuda a apresurar la convergencia, ya que todas las capas estan alrededor de 0

- ReLU: Función de activación $ReLU(z) = max(0,z)$, en la cual, se activa para cuando el valor de la neurona es $>= 0$