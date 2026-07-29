# Training Deep Neural Nets

## El problema de los gradientes que explotan/desaparecen

La segunda fase del algoritmo de "backprop" funciona propagando el gradiente del error de capa de salida a capa de entrada y una vez que se ha calculado la funcion de perdida, se utilizan para ctualizar los parámetros de la red.

Desgraciadamente, los gradientes suelen hacerse más y más pequeños cuanto desciende en la red, teniendo como resultado que la actualización de las primeras capas se ven poco afectadas por el cambio. A esto se le conoce como el problema de los gradientes que desaparecen.

Por otro lado, puede suceder los contrario con los gradientes. Estos crecen y crecen hasta que las capas tienen unos pesos absolutamente enormes. A esto se le conoce como el problema de los gradientes que explotan.

Como se pudo demostrar en un paper academico [ https://homl.info/47 ], el problema con la inestabilidad de los pesos era una combinación entre la función sigmoide de activación y el metodo de inicialización. En el caso de la función sigmoide, cuando las entradas son muy grandes ($[ - \infty, \infty ]$) la funcion de activación tiende a saturar $[ 0, 1 ]$, con una derivada que tiende a $0$, con lo que la backprop no tiene casi gradiente que propagar.

### Inicialización de Glorot e Inicialización de He

En el paper anterior, Glorot y Bengio proponen un método para aliviar significativamente la inestabilidad de los gradientes señalando que la señal tiene que viajar correctamente en ambas direcciones: hacia delante cuando se hacen predicciones y hacia detras cuando se propagan los gradientes. Entonces, los autores explican que se necesita que la varianza de las salidas y entradas sean iguales, asi como que los gradientes tienen que tener una varianza igual antes y despues de propagarse. Esto no es posible si no existen las mismas neuronas de entrada y salida en una capa (fan-in y fan-out). La estrategia que propusieron fue inicializar los pesos de manera que $\large fan_{avg} = (fan_{in} + fan_{out}) / 2$:

- Distribución normal con media 0 y varianza $\large \sigma ^2 = \frac{1}{fan_{avg}}$

- Distribución uniforme entre $\large [ -r, r, ]$ con $r = \sqrt{\frac{3}{fan_{avg}}}$

Por otro lado, la estrategia de inicialización propuesta para la función ReLU de activación son las Inicializaciones de He o Kaiming

### Mejores funciones de activación

Aun que la función de activación sigmoide es parecida a la de nuestras neuronas, se ha demostrado que la ReLU es mucho mejor para las DDNN, debido a su fácil cálculo y que no satura los valores por muy altos/bajos que sean. Esta última función si bien es mejor, no es perfecta, ya que sufre de un problema por el cual algunas neuronas "mueren" (dejan de calcular). Esta "muerte" hace que los pesos se trastoquen tanto que el input de la función será siempre negativo.

Para resolver este problema hay variaciones de la funcion:

#### Leaky ReLU

Esta función esta caracterizada por $LeakyReLU_{\alpha}(z) = max(az, z)$, donde el hiperparametro $\alpha$ define cuanto "filtra" la función, preparando una pendiente en la función con el que dar valores a las entradas negativas.

En un paper de 2015 [ https://homl.info/49 ] se comparan variantes de ReLU y se obtiene un mejor desempeño en todas, sobre la función original. En este experimento se explica que para Leaky ReLU funciona incluso mejor pendientes altas que mas bajas (0.2 vs 0.01). Por otro lado se evalua RReLU (random ReLU), en el que se elige $\alpha$ de manera aleatoria en un rango dentro del entrenamiento y se ajusta a la media para el testeo. Por ultimo se estudia PReLU (parametric leaky ReLU), donde el $\alpha$ se convierte en un parametro mas del entrenamiento y es sujeto a backprop. Este último es capaz de tasas de acierto muy buenas en grandes conjuntos de imagenes, pero cuando la información escasea tiende a hacer overfitting

#### ELU y SELU

En un paper de 2015 [ https://homl.info/50 ] se planteo una función de activación que superaba el desempeño de todas las variantes de ReLU, reduciendo tiempo de entrenamiento y mejorando las predicciones de la red neuronal en el conjunto de prueba. La ecuación que describe la Exponential Linear Unit (ELU) es la siguiente:

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

Por otro lado, en 2017 se planteó una variación de ELU, SELU (scaled ELU) en un paper [ https://homl.info/selu ]. En dicho paper se demostro que si se implementa una red neuronal de capas densas y si todas las capas utilizan SELU, la red normalizará por si misma las salidas con una media de 0 y una desviación estandard de 1, lo cual contrarresta completamente la explosión/desvanecimiento de los gradiente.

Para utilizarlo se ha de llamar al modulo nn.SELU, con las siguientes restricciones:

- Las entradas deben estar normalizadas, con una media de 0 y desviación 1

- Todos los pesos de las capas ocultas tienen que estar inicializadas con el metodo de LeCun

- La auto normalización no esta garantizada con arquitecturas que no sean MLPs llanas

- No se pueden utilizar tecnicas de regularización como ℓ1 o ℓ2, normalización de lotes, capas, max o dropout

#### GELU, Swish, SwiGLU, Mish, y RELU²

La función GELU(Gaussian Error Linear Unit) fue presentada en un paper de 2016 [ https://homl.info/gelu ] y se puede considerar una variante suave de ReLU. Se define con la siguiente fórmula:
$$
\large
GELU(z) = z\Phi(z)
$$

Siendo $\Phi$ una función de distribución acumilativa Gaussiana $(CDF): \Phi(z)$, que corresponde a la probabilidad de que un valor muestreado al azar de una distribución normal con mediana 0 y varianza 1 es menor que z.

Swish nace como una variación de GELU en un paper de 2017 [ https://homl.info/swish ], una aproximación de la función de activación para aligerar el tiempo de computación de los gradientes. Esta se ve definida como:

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

Una variante popular de Swish es SwiGLU [ https://homl.info/swiglu ], en la cual las entradas pasan por la función de activación Swish y en paralelo por una capa lineal para, finalmente, multiplicar los resultados:

$$
\large
SwiGLU(z) = Swish_{\beta}(z) ⊗ Linear(z)
$$

Otra función de activación parecida a GELU es Mish, planteada en 2019 [ https://homl.info/mish ]. Se define de la siguiente manera:

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

### Normalización de lotes

Si bien la inicialización de Kaiming conjuntamente con ReLU evita los problemas anteriores en el principio del entrenamiento, eso no quiere decir que no vaya a volver a lo largo del proceso

Dado lo anterior, en un paper de 2015 [ https://homl.info/51 ] se planteo la normalización de lotes (BN), la cual consistia en añadir una operación antes/despues del cálculo de la función de activación, normalizando y centrando en 0 cada una de las entradas para despues escalar y mover el resultado utilizando dos vectores de parameotrs por capa; uno para escalado, otro para el movimiento.

Con el objetivo de hacer lo anterior, el algoritmo necesita aproximar la media y la desviación de cada entrada, evaluando estas variables sobre el lote actual, de la siguiente manera:

$$
\large
\begin{align}
1. \quad & \boldsymbol{\mu}_B = \frac{1}{m_B} \sum_{i=1}^{m_B} \mathbf{x}^{(i)} \\[ 10pt ]
2. \quad & \boldsymbol{\sigma}_B^2 = \frac{1}{m_B} \sum_{i=1}^{m_B} \left( \mathbf{x}^{(i)} - \boldsymbol{\mu}_B \right)^2 \\[ 10pt ]
3. \quad & \widehat{\mathbf{x}}^{(i)} = \frac{\mathbf{x}^{(i)} - \boldsymbol{\mu}_B}{\sqrt{\boldsymbol{\sigma}_B^2 + \varepsilon}} \\[ 10pt ]
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

### Normalización de capas

La normalización de capas (LN) es parecida a la de lotes, pero en vez de normalizar a lo largo de la dimension del lote, lo hace en la dimensión de las característcias. Así se presentó en el paper de 2016 [ https://homl.info/layernorm  ].

Una de las ventajas que tiene LN es que puede calcular en el momento las estadísticas independientemente de cada instancia, con lo que se comporta igual en entrenamiento como en test. Este método también aprende una escala y un offset para cada característica de entrada. 

### Recorte de gradiente

El recorte de gradiente es una técnica que se utiliza para evitar la explosión de gradientes. Se basa en definir un umbral que no pueden sobrepasar en el momento de backrpop. Esta técnica se suele utilizar en las redes neuronales recurrentes, ya que la normalización de lotes es complicada.

## Reusar capas preentrenadas

Normalmente no es la mejor idea entrenar una red profunda desde cero sin buscar otra que ya intente resolver el mismo problema, ya que en el caso de encontrarla se pueden reusar la mayoria de sus capas, exceptuando las de arriba del todo. A esta técnica se le conoce como "transfer learning" y no solo acelerará el entrenamiento considerablemente, si no que necesitará menos información de entrenamiento.

En el caso de tener acceso a una red profunda entrenada para clasificar imagenes en varias clases, incluidas imagenes de coches, y tu idea es implementar una red que clasifique diferentes tipos de coches. Ya que estas tareas son similares, podemos intentar reutilizar partes de la primera red para entrenar la nueva. Una regla general util es que cuanto mas se parezcan las tareas a resolver, mas cantidad de capas preentrenadas se podrán utilizar, aun que la capa de salida habra que reemplazarla y las capas más altas también, ya que son las que estan entrenadas en los detalles más finos. 

> Cabe destacar que si la entrada del nuevo modelo no es del mismo tamaño, será necesario aplicar un paso previo de preprocesado para reescalar ese tamaño.Para poner un ejemplo, si la red neuronal está entrenada con imágenes de teléfonos móviles, servirá para hacer inferencia sobre otras imágenes tomadas con teléfonos, pero no para aquellas adquiridas con satélites.

El objetivo es encontrar el número correcto de capas a reutilizar. Se puede empezar "congelando" (requires_grad=False) las capas para no modificarlas y entrenar el modelo para ver su desempeño. Mas tarde se pueden descongelar una o dos y volver a entrenar para ver si mejora. En general, cuanta mas información tienes, mas capas puedes descongelar, aun que el objetivo es reutilizar capas para acelerar el proceso de entrenamiento.

### Preentrenamiento no supervisado

En el caso de tener una tarea compleja, no disponer de un modelo preentrenado y no tener mucha información etiquetada, no poder obtener mas, existe la posibilidad de aplicar el preentrenamiento no supervisado. Si existe un conjunto abundante de información no etiquetada, se puede utilizar para entrenar un modelo no supervisado (así como un autoencoder) y más tarde reutilizar las capas inferiores de este modelo, añadir una capa de salida y hacer fine-tune en la red resultante utilizando información ya etiquetada

Actualmente se suele entrenar el modelo completo sobre la información no etiquetada de una sentada, en vez de ir entrenando capa por capa, congelando las anteriores para entrenar nuevas, ademas de utilizar modelos como los autoencoders o modelos de difusion más que las maquinas de Boltzmann

### Preentrenamiento en una tarea auxiliar

Una última opcion es entrenar una primera red neuronal en una tarea auxiliar, de la cual se puede obtener o generar facilmente información etiqutada, para despues reutilizar las capas inferiores para la tarea para la cual quieres implementar un modelo. Normalmente las capas inferiores del primer modelo aprenderan a reconocer patrones que serán útiles en la segunda red neuronal.

Por ejemplo:
> Se quiere implementar un sistema de reconocimiento de rostros, únicamente con 2 o 3 imágenes por individuo. Ya que obtener cientos de imágenes por persona no es plausible, se podría utilizar un conjunto de imágenes público (como VGGFace2) con millones de rostros y entrenar una primera red neuronal con estas imágenes, para detectar la misma persona en dos retratos diferentes. Este preentrenamiento será muy útil para más tarde reutilizar las capas del primer modelo, que ya hayan aprendido a reconocer patrones, y entrenar este segundo modelo con la información escasa.

## Optimizadores más rapidos

Hasta ahora hemos visto 4 métodos de acelerar el entrenamiento de las redes neuronales profundas:
- Estratégias de inicialización
- Funciones de activación
- Normalización de lotes y capas
- Reutilización de capas

Ahora veremos los varios optimizadores mas allá del optimizador de descenso de gradiente normal.

### Momentum

La idea de la optimización del momentum presentada en un paper de 1964 [ https://homl.info/54 ] se basa en afectar el gradiente calculado con una variable dentro del vector de momentum $m$ multiplicado por LR. Hasta ahora el descenso de gradiente se basaba en actualizar los pesos restando directamente el gradiente de la funcion de coste respecto a los pesos multiplicada por el LR de la siguiente manera:
$$
\large
\theta \leftarrow \theta - \eta \nabla_{\theta}J(\theta)
$$

En otras palabras, en la optimización del momentum, el gradiente es usado como una fuerza de aceleración, no como una velocidad, con lo que para evitar que aumente indefinidamente, se implementa un "mecanismo de rozamiento" en forma de hiperparametro $\beta$ conocido como coeficiente de momentum con valores entre $[0,1]$ (0 máxima fricción, 1 sin fricción). Con lo que la ecuación que define este momentum es la siguiente:

$$
\large
\begin{align}
1.  \quad & m \leftarrow \beta m - \eta \nabla_{\theta}J(\theta) \\ 
2.  \quad & \theta \leftarrow \theta m 
\end{align}
$$

Ya que en la práctica los gradientes no son constantes, puede no apreciarse tanto la aceleración del entrenamiento, pero esta optimización escapa mucho antes de las mesetas que el descenso de gradiente normal, además de ayudar a evitar los máximos locales.

### Gradiente acelerado de Nesterov

Yurii Nesterov propuso en su paper de 1983 [ https://homl.info/55 ] una variante a la optimización de momentum, el gradiente acelerado de Nesterov (NAG), la cual mide el gradiente de la función de coste mas alla de la dirección del momentum, en $\theta+\beta m$, quedando definida de la siguiente manera:

$$
\large
\begin{align}
1. & \quad m \leftarrow \beta m - \eta \nabla_{\theta}J(\theta+\beta m) \\ 
2. & \quad \theta \leftarrow \theta + m 
\end{align}
$$

Esta modificación funciona ya que normalmente el gradiente apuntara en la dirección correcta, con lo que será aún más correcto usar el gradiente medido mas alla en esa dirección, en vez del gradiente en la posición original.

### AdaGrad

Considerando el problema del cuenco alargado en el que el descenso de gradiente comienza yendo rapidamente por la pendiente mas acentuada, aun que no sea la que conduzca al mínimo global, para despues moverse lentamente valle abajo. El algoritmo Adagrad [ https://homl.info/56 ] es capaz de corregir la dirección con el objetivo de moverse hacia el punto óptimo global antes, con una definición como la siguiente:

$$
\large
\begin{align}
1. & \quad s \leftarrow s + \nabla_{\theta}J(\theta) ⊗ \nabla_{\theta}J(\theta)\\ 
2. & \quad \theta \leftarrow \theta - \eta \nabla_{\theta}J(\theta) ⊘ \sqrt{s + \epsilon}
\end{align}
$$

> El símbolo $⊗$ representa multiplicación elemento por elemento, mientras que $⊘$ representa división elemento por elemento

El primer paso del algorítmo acumula el cuadrado de los gradientes en el vector s, lo cual es equivalente a calcular $s_i \leftarrow s_i + (\frac{\partial j(\theta)}{\partial \theta_i})²$ para cada elemento $s_i$ del vector s.

El segundo paso es casi identico al descenso de gradiente, con una diferencia notable ya que el vector de gradientes es escalado por un factor $\sqrt{s + \epsilon}$, siendo $\epsilon$ un factor para evitar las divisiones por 0. Esta forma vectorial es equivalente a calcular para todos los parametros $\theta_i$ lo siguiente: $\large \theta_i \leftarrow \theta_i  \frac{\frac{\eta \partial j(\theta)}{\partial\theta_i}}{\sqrt{s }+ \epsilon}$

Este algoritmo hace decaer el LR, pero lo hace mucho más rapido para dimensiones empinadas que para cuestas suaves, lo que se conoce como aprendizaje adaptativo. 

Por otro lado, si bien funciona correctamente para problemas cuadraticos, puede parar muy pronto el entrenamiento debido a que escala tanto el LR que es incapaz de llegar al punto óptimo global

### RMSProp

RMSProp es una variante de adagrad que evita que el entrenamiento pare antes de converger dado a la desaparición del LR acumulando solo los gradientes de las iteraciones mas recientes, en vez de todos desde el inicio del entrenamiento. Para esto se utiliza la desintegración exponencial en el primer paso:

$$
\large
\begin{align}
1. & \quad  s \leftarrow \alpha s + (1 - \alpha) \nabla_{\theta}J(\theta) ⊗ \nabla_{\theta}J(\theta)\\ 
2. & \quad \theta \leftarrow \theta - \eta \nabla_{\theta}J(\theta) ⊘ \sqrt{s + \epsilon}
\end{align}
$$

> Siendo $\alpha$ el hiperparametro del ratio de desintegración, normalmente definido a 0.9

### Adam

Adam, que significa estimacion de momento adaptativa, combina las ideas de la optimización del momentm y el algoritmo RMSProp. Este optimizador tiene en cuenta la media de la descomposición exponencial de gradientes (optimización de momentum), así como lleva la cuenta de las medias de desintegración exponencial de gradientes cuadrados pasados. Estas son estimaciones de la media sin centrar (primer momentum) y la varianza de los gradientes (segundo momentum):

$$
\large
\begin{align}
1. & \quad\mathbf{m} &\leftarrow \beta_1 \mathbf{m} - (1-\beta_1)\,\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) \\
2. & \quad\mathbf{s} &\leftarrow \beta_2 \mathbf{s} + (1-\beta_2)\,\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) \otimes \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) \\
3. & \quad\hat{\mathbf{m}} &\leftarrow \frac{\mathbf{m}}{1-\beta_1^t} \\
4. & \quad\hat{\mathbf{s}} &\leftarrow \frac{\mathbf{s}}{1-\beta_2^t} \\
5. & \quad\boldsymbol{\theta} &\leftarrow \boldsymbol{\theta} + \eta\,\hat{\mathbf{m}} \oslash \sqrt{\hat{\mathbf{s}}+\varepsilon}
\end{align}
$$

> Siendo $t$ el número de la iteración, $\beta_1$ sería el momentum y $\beta_2$ correspondria al ratio de desintegración $\alpha$ de RMSProp

Mientras que los pasos 1, 2 y 5 son muys similares a los de ADAM y RMSProp, los pasos 3 y 4 son definiciones de los valores para $m$ y $s$, ya que al inicializarse a $0$ estarían sesgados por ese valor al principio del entrenamiento

#### AdaMax

Adam acumula los cuadrados de los gradientes en $s$, como hemos visto antes, además de escalar por debajo las actualizaciones de los parametros por la raiz de $s$, es decir, Adam escala las actualizaciones por la norma $l_2$ (la raíz cuadrada de la suma de los cuadrados)

Por otro lado, AdaMax, introducido en el mismo paper que Adam, reemplaza $l_2$ por $l_{\infty}$, en concreto reemplaza el paso 2 del algoritmo con 
$$
\large
s \leftarrow max(\beta_2 s, abs (\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})))
$$
elimina el paso 4 y en el paso 5 escala la actualización de gradientes por $s$; el máximo del valor absoluto de los gradientes desintegrados en el tiempo.

En definitiva, esta variante puede hacer que el algorítmo sea más estable que Adam, pero depende bastante del conjunto de información y en general Adam tiene un mejor desempeño.

#### NAdam

Esta variante es la optimización de Adam con el truco de Nesterov añadido, con lo que llegará a la convergencía más rápido que el algorítmo de Adam base.

#### AdamW

Esta es otra variante de Adam que integra la técnica de regularización de "desintegración de pesos". Esa reduce el tamaño de los pesos de los modelos en cada iteración multiplicandolos por un factor de desintegración

### Conclusión 

Todas las técnicas que se han visto dependen de las Jacobianas, que miden la pendiente de la función de pérdida, aunque hay más algorítmos de optimización basados en las Hessianas, midiendo como cambianlas Jacobianas a lo largo de cada eje. El problema de las Hessianas es el coste computacional  y de espacio de memoria de las segundas derivadas parciales, ya que existen $n²$ por cada salida.

Para acabar, se resume en la siguiente tabla todas las técnicas de optimización con una puntuación del 1-3 en su velocidad de convergencia y la calidad de esta

| Class                                | Convergence speed (1-3) | Convergence quality (1-3) |
|--------------------------------------|--------------------------|----------------------------|
| SGD                                  | 1                        | 3                          |
| SGD(momentum=...)                    | 2                        | 3                          |
| SGD(momentum=..., nesterov=True)     | 2                        | 3                          |
| Adagrad                              | 2                        | 1 *(stops too early)*      |
| RMSprop                              | 2                        | 2 o 3                      |
| Adam                                 | 2                        | 2 o 3                      |
| AdaMax                               | 2                        | 2 o 3                      |
| NAdam                                | 2                        | 2 o 3                      |
| AdamW                                | 2                        | 2 o 3                      |

## Programación de la tasa de aprendizaje
