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
## Redes neuronales
### a) Modelo básico

La manera en la que funcionan las neuronas es la siguiente:

1. Entra un input (Xn), que se multiplica por un peso (Wn).
2. Se hace el sumatorio de las respuestas a esos inputs.
3. Si el resultado supera un umbral (T), la neurona se activa y transmite información.

[![Red neuronal básica](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/raw/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020260326121948.png)](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/blob/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020260326121948.png)

Este modelo deja varios conceptos clave:

- Comportamiento binario (_all or none_)
- Influencia acumulativa
- Pesos sinápticos

> Cabe resaltar que es un modelo representativo de las neuronas cerebrales, pero no completamente cierto, ya que quedan muchos interrogantes por responder.

Si formamos una red de neuronas dependientes, la función que define la salida (Z) es:

Z→=f(X→,W→,T→)

La salida depende de la entrada, los pesos y los umbrales que se definan. Esto convierte a una red neuronal en una gran función para aproximar resultados.

### Función de precisión

Una forma de evaluar la precisión de la red puede ser:

P=−‖D→−Z→‖2

donde:

- (D→) es el resultado deseado
- (Z→) es el resultado obtenido

Dado que el objetivo es aproximar el resultado predicho al deseado, si no coinciden, se deben modificar los pesos y/o umbrales con el objetivo de maximizar la función de desempeño.

### Ajuste de pesos

[](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/blob/main/Apuntes%20ML/README.md#ajuste-de-pesos)

Para recalcular los pesos, podemos utilizar una función con derivadas parciales, como la siguiente:

Δw→=r(∂P∂w1i+∂P∂w2j)

donde:

- (r) es una constante que decide la amplitud del movimiento de los pesos
- (i) y (j) son coordenadas en un plano

Esto nos deja una función de ascenso de gradiente.

### Adaptación a función continua

[](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/blob/main/Apuntes%20ML/README.md#adaptaci%C3%B3n-a-funci%C3%B3n-continua)

Como el ascenso de gradiente necesita un dominio continuo y la función que define la salida es discreta, se propuso utilizar la siguiente función eliminando el umbral “escalón”:

Z→′=f′(X→,W→)

Para el umbral se añadirá un nuevo input (w_0) que siempre será (-1) e igualará al umbral, lo cual permitirá que cualquier resultado del producto de una entrada por un peso mayor que 0 active la neurona.

Para adaptarlo al dominio que buscamos, convertimos nuestro umbral escalón en uno **sigmoide**:

T=11+e−α

que, con valores de (α) que tiendan a (∞), tiende asintóticamente a 1, y con valores cercanos a (−∞), tiende a 0.

[![Función sigmoide](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/raw/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020260326155814.png)](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/blob/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020260326155814.png)

### Red neuronal simple

El proceso de la red neuronal más sencilla es el siguiente:

[![Red neuronal simple](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/raw/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020260326160124.png)](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/blob/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020260326160124.png)

Para este ejemplo, y teniendo en cuenta la función de desempeño definida, se generan las siguientes derivadas parciales para calcular la influencia de los pesos en la transformación de la información de entrada.

[![Derivadas parciales](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/raw/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020260326162504.png)](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/blob/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020260326162504.png)

### Escalado

Si esta red la duplicamos como en la siguiente figura, no explotaría de manera exponencial el número de cálculos que se necesita para evaluarla, ya que podríamos reutilizar los cálculos entre redes, aunque estas puedan influenciarse entre ellas.

[![Red escalada](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/raw/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020260326162745.png)](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/blob/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020260326162745.png)

### Conclusiones

1. Las redes son lineales en profundidad: el coste computacional requerido aumentará de manera lineal con el aumento de la profundidad de la red.
2. El cálculo necesario estará proporcionalmente relacionado con las conexiones posibles.
