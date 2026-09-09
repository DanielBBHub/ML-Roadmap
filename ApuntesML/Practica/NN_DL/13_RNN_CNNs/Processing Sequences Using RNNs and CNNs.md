# Processing Sequences Using RNNs and CNNs

## Capas y neuronas recurrentes

Hasta ahora nos hemos concentrado en las redes que se movian en un único sentido, hacia delante. Estas neuronas recurrentes pueden transmitir información pero tambien tienen conexiones apuntando hacia atras

Si nos fijamos en laa red neuronal recurrente mas simple, la que esta compuesta de una neurona, podemos ver que esta esta recibiendo una entrada, generando una salida y enviando esta salida de vuelta a ella misma. En cada punto en el tiempo $t$, esta neurona recurrente recibe una entrada $x_{(t)}$, así como su propia salida anterior, $y_{(t-1)}$ (si es el primer paso y no hay una salida anterior se suele definir a $0$). En el caso de formar una capa recurrente, cada paso $t$ la capa recibe un vector de entrada $x_{(t)}$, así como un vector de la salida anterior $y_{(t-1)}$

Cada neurona recurrente tiene dos pares de pesos: una para las entradas y otras para las salidas anteriores; $w_x$ y $w_y$ o $W_x$ y $W_y$ cuando tratamos con una capa recurrente (escalares para neuronas, vectores para capas)

El vector de salida de estas capas se define de la siguiente manera:

$$
\large
y_{(t)} = \phi (W_x^t x_{(t)} + W_y^t y_{(t-1)} + b)
$$

