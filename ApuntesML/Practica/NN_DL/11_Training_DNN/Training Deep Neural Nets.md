# Training Deep Neural Nets

## El problema de los gradientes que explotan/desaparecen

La segunda fase del algoritmo de "backprop" funciona propagando el gradiente del error de capa de salida a capa de entrada y una vez que se ha calculado la funcion de perdida, se utilizan para ctualizar los parámetros de la red.

Desgraciadamente, los gradientes suelen hacerse más y más pequeños cuanto desciende en la red, teniendo como resultado que la actualización de las primeras capas se ven poco afectadas por el cambio. A esto se le conoce como el problema de los gradientes que desaparecen.

Por otro lado, puede suceder los contrario con los gradientes. Estos crecen y crecen hasta que las capas tienen unos pesos absolutamente enormes. A esto se le conoce como el problema de los gradientes que explotan.

Como se pudo demostrar en un paper academico [https://homl.info/47], el problema con la inestabilidad de los pesos era una combinación entre la función sigmoide de activación y el metodo de inicialización. En el caso de la función sigmoide, cuando las entradas son muy grandes ($[- \infty, \infty]$) la funcion de activación tiende a saturar $[0, 1]$, con una derivada que tiende a $0$, con lo que la backprop no tiene casi gradiente que propagar.

### Inicialización de Glorot e Inicialización de He

En el paper anterior, Glorot y Bengio proponen un método para aliviar significativamente la inestabilidad de los gradientes señalando que la señal tiene que viajar correctamente en ambas direcciones: hacia delante cuando se hacen predicciones y hacia detras cuando se propagan los gradientes. Entonces, los autores explican que se necesita que la varianza de las salidas y entradas sean iguales, asi como que los gradientes tienen que tener una varianza igual antes y despues de propagarse. Esto no es posible si no existen las mismas neuronas de entrada y salida en una capa (fan-in y fan-out). La estrategia que propusieron fue inicializar los pesos de manera que $\large fan_{avg} = (fan_{in} + fan_{out}) / 2$:

- Distribución normal con media 0 y varianza $\large \sigma ^2 = \frac{1}{fan_{avg}}$

- Distribución uniforme entre $\large [-r, r,]$ con $r = \sqrt{\frac{3}{fan_{avg}}}$

Por otro lado, la estrategia de inicialización propuesta para la función ReLU de activación son las Inicializaciones de He o Kaiming

### Mejores funciones de activación

Aun que la función de activación sigmoide es parecida a la de nuestras neuronas, se ha demostrado que la ReLU es mucho mejor para las DDNN, debido a su fácil cálculo y que no satura los valores por muy altos/bajos que sean. Esta última función si bien es mejor, no es perfecta, ya que sufre de un problema por el cual algunas neuronas "mueren" (dejan de calcular). Esta "muerte" hace que los pesos se trastoquen tanto que el input de la función será siempre negativo.

Para resolver este problema hay variaciones de la funcion:

#### Leaky ReLU

Esta función esta caracterizada por $LeakyReLU_{\alpha}(z) = max(az, z)$, donde el hiperparametro $\alpha$ define cuanto "filtra" la función, preparando una pendiente en la función con el que dar valores a las entradas negativas.

En un paper de 2015 [https://homl.info/49] se comparan variantes de ReLU y se obtiene un mejor desempeño en todas, sobre la función original. En este experimento se explica que para Leaky ReLU funciona incluso mejor pendientes altas que mas bajas (0.2 vs 0.01). Por otro lado se evalua RReLU (random ReLU), en el que se elige $\alpha$ de manera aleatoria en un rango dentro del entrenamiento y se ajusta a la media para el testeo. Por ultimo se estudia PReLU (parametric leaky ReLU), donde el $\alpha$ se convierte en un parametro mas del entrenamiento y es sujeto a backprop. Este último es capaz de tasas de acierto muy buenas en grandes conjuntos de imagenes, pero cuando la información escasea tiende a hacer overfitting

### ELU y SELU

En un paper de 2015 [https://homl.info/50] se planteo una función de activación que superaba el desempeño de todas las variantes de ReLU, reduciendo tiempo de entrenamiento y mejorando las predicciones de la red neuronal en el conjunto de prueba. La ecuación que describe la Exponential Linear Unit (ELU) es la siguiente:

$$
\text{ELU}_{\alpha}(z) = 
\begin{cases} 
\alpha(\exp(z) - 1) & \text{if } z < 0 \\ 
z & \text{if } z \geq 0 
\end{cases}
$$

Aun que el resultado de esta función de activación sea parecido al de la ReLU, tiene varios puntos en los que se diferencian:

- Devuelve valores negativos para $z<0$, lo cual alivia el problema del desvanecimiento de gradientes. El hiperparametro $\alpha$ define los opuesto al valor que devuelve la función cuando $z$ es un número negativo grande.

- Evita el problema de neuronas muertas, ya que tiene un valor $\neq 0$ para $z<0$

- Si $\alpha = 1$, la funcion es suave en cualquier punto, lo cual ayuda a acelerar el calculo del descenso de gradiente

Para utilizar ELU en PyTorch solo hay que utilizar el modulo nn.ELU junto a la inicialización Kaiming

Por otro lado, en 2017 se planteó una variación de ELU, SELU (scaled ELU) en un paper [https://homl.info/selu]. En dicho paper se demostro que si se implementa una red neuronal de capas densas y si todas las capas utilizan SELU, la red normalizará por si misma las salidas con una media de 0 y una desviación estandard de 1, lo cual contrarresta completamente la explosión/desvanecimiento de los gradiente.

Para utilizarlo se ha de llamar al modulo nn.SELU, con las siguientes restricciones:

- Las entradas deben estar normalizadas, con una media de 0 y desviación 1

- Todos los pesos de las capas ocultas tienen que estar inicializadas con el metodo de LeCun

- La auto normalización no esta garantizada con arquitecturas que no sean MLPs llanas

- No se pueden utilizar tecnicas de regularización como ℓ1 o ℓ2, normalización de lotes, capas, max o dropout

### GELU, Swish, SwiGLU, Mish, y RELU²

La función GELU(Gaussian Error Linear Unit) fue presentada en un paper de 2016 [https://homl.info/gelu] y se puede considerar una variante suave de ReLU. Se define con la siguiente fórmula:
$$
\large
GELU(z) = z\Phi(z)
$$

Siendo $\Phi$ una función de distribución acumilativa Gaussiana $(CDF): \Phi(z)$, que corresponde a la probabilidad de que un valor muestreado al azar de una distribución normal con mediana 0 y varianza 1 es menor que z.

Swish nace como una variación de GELU en un paper de 2017 [https://homl.info/swish], una aproximación de la función de activación para aligerar el tiempo de computación de los gradientes. Esta se ve definida como:

$$
\large
Swish(z)= z\sigma(z)
$$

En el mismo paper se presenta la generalización de Swish añadiedo un parametro $\beta$ para escalar la entrada de la función de activacion:

$$
\large
Swish_{\beta}(z)= z\sigma(\beta z)
$$

Este parametro puede ser entrenable, normalmente suele tener un único parametro para el modelo entero o, como mucho, uno por cada capa para mantener el modelo eficiente y evitar el overfitting

Una variante popular de Swish es SwiGLU [https://homl.info/swiglu], en la cual las entradas pasan por la función de activación Swish y en paralelo por una capa lineal para, finalmente, multiplicar los resultados:

$$
\large
SwiGLU(z) = Swish_{\beta}(z) ⊗ Linear(z)
$$

Otra función de activación parecida a GELU es Mish, planteada en 2019 [https://homl.info/mish]. Se define de la siguiente manera:

$$
\large
mish(z) = ztanh(sofplus(z))
$$

donde
$$
\large
sofplus(z) = log(1 + exp(z))
$$

Asi como GELU y Swish, es una variante de ReLU suave, no convexa y no monotona

## Normalización de lotes

Si bien la inicialización de Kaiming conjuntamente con ReLU evita los problemas anteriores en el principio del entrenamiento, eso no quiere decir que no vaya a volver a lo largo del proceso

Dado lo anterior, en un paper de 2015 [https://homl.info/51] se planteo la normalización de lotes (BN), la cual consistia en añadir una operación antes/despues del cálculo de la función de activación, normalizando y centrando en 0 cada una de las entradas para despues escalar y mover el resultado utilizando dos vectores de parameotrs por capa; uno para escalado, otro para el movimiento.

Con el objetivo de hacer lo anterior, el algoritmo necesita aproximar la media y la desviación de cada entrada, evaluando estas variables sobre el lote actual, de la siguiente manera:

$$
\large
\begin{align}
1. \quad & \boldsymbol{\mu}_B = \frac{1}{m_B} \sum_{i=1}^{m_B} \mathbf{x}^{(i)} \\[10pt]
2. \quad & \boldsymbol{\sigma}_B^2 = \frac{1}{m_B} \sum_{i=1}^{m_B} \left( \mathbf{x}^{(i)} - \boldsymbol{\mu}_B \right)^2 \\[10pt]
3. \quad & \widehat{\mathbf{x}}^{(i)} = \frac{\mathbf{x}^{(i)} - \boldsymbol{\mu}_B}{\sqrt{\boldsymbol{\sigma}_B^2 + \varepsilon}} \\[10pt]
4. \quad & \mathbf{z}^{(i)} = \boldsymbol{\gamma} \otimes \widehat{\mathbf{x}}^{(i)} + \boldsymbol{\beta}
\end{align}
$$

Siendo, en este algoritmo:
- $\mu_B$: el vector con la media de las entradas sobre el lote $\beta$
- $m_B$: el numero de instancias en el vector
- $x^{(i)}$: el vector de entrada del lote para la instancia $i$
- $\sigma_B^2$: el vector con las desviaciones sobre el lote $\beta$
- $\widehat{\mathbf{x}}^{(i)}$: el vector de entradas normalizado y centrado en $0$ para la instancia $i$
- $\varepsilon$: termino suavizante, para evitar las divisiones por $0$ y que los gradientes no exploten
- $\gamma$: el vector del parametro de escalado de la salida para la capa
- $⊗$: representa la multiplicación, elemento a elemento
- $\beta$: el movimiento del vector de parametros de salida
- $\mathbf{z}^{(i)}$: es la salida de la operacion BN (la version de la entrada rescalada y movida)

En definitiva, hay 4 vectores de parametros aprendidos en cada capa BN:
- Aprendidos por backprop:
    - $\gamma$
    - $\beta$
- Aproximados utilizando media exponencial movil:
    - $\mu$
    - $\sigma^2$

Es necesario resaltar que los dos últimos vectores de parámetros se aprenden en el entrenamiento, pero solo son usados una vez este acaba. En el momento en el que utilizamos model.eval(), estos cambian por $\mu \rightarrow \mu_B$ y $\sigma^2 \rightarrow \sigma_B^2$

Estos avances permitieron grandes mejoras en los modelos de clasificación de imágenes de los investigadores; se evito en gran medida el desvanecimiento de gradientes, hasta el punto de poder utilizar funciones de activación que saturan, ademas de ser menos sensibles a la inicialización de pesos e incluso pudieron utilizar LR mas altos. Finalmente, esta técnica actua tambien como regularizador, reduciendo la necesidad de otras técnicas.

Esta estratégia añade complejidad al modelo, volviéndolo mas lento en la inferencia, aun que es posible fusionar las capas BN con las capas previas despues de entrenar, pudiendo evitar esta penalización de tiempo de ejecución.