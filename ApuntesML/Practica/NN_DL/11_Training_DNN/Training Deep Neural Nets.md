# Entrenando redes neuronales profundas

## El problema de los gradientes que explotan/desaparecen

La segunda fase del algoritmo de backpropagation funciona propagando el gradiente del error desde la capa de salida hasta la capa de entrada. Una vez que se ha calculado la función de pérdida, se utilizan para actualizar los pesos de la red.

Desgraciadamente, los gradientes suelen hacerse cada vez más pequeños conforme descienden en la red. Esto tiene como resultado que la actualización de las primeras capas sea muy pequeña, ralentizando significativamente el entrenamiento. A esto se le conoce como el problema de la **desaparición de gradientes** (vanishing gradients).

Por otro lado, puede suceder lo contrario. Los gradientes crecen exponencialmente hasta que las capas tienen unos pesos absolutamente enormes. A esto se le conoce como el problema de la **explosión de gradientes** (exploding gradients).

En un paper académico [[47]](https://homl.info/47), se demostró que el problema con la inestabilidad de los pesos era una combinación entre la función sigmoide de activación y el método de inicialización de pesos.

---

### Inicialización de Glorot e Inicialización de He

En el paper anterior, Glorot y Bengio proponen un método para aliviar significativamente la inestabilidad de los gradientes. La idea es que la señal tiene que viajar correctamente en ambas direcciones (forward y backward).

**Inicialización de Glorot** (también llamada Xavier):
- Distribución normal con media 0 y varianza $\large \sigma ^2 = \frac{1}{fan_{avg}}$
- Distribución uniforme entre $\large [ -r, r ]$ con $r = \sqrt{\frac{3}{fan_{avg}}}$

Por otro lado, la estrategia de inicialización propuesta para la función ReLU de activación son las **Inicializaciones de He** (o Kaiming), que escalan mejor con redes profundas.

---

### Mejores funciones de activación

Aunque la función de activación sigmoide es parecida a la de nuestras neuronas biológicas, se ha demostrado que la ReLU es mucho mejor para las DNN, debido a su fácil cálculo y a que no satura los valores en comparación con la sigmoide.

Sin embargo, la ReLU presenta el problema de las "neuronas muertas": si un neurón recibe un valor negativo, siempre devolverá 0 y sus pesos nunca se actualizarán.

Para resolver este problema hay variaciones de la función:

#### Leaky ReLU

Esta función está caracterizada por $LeakyReLU_{\alpha}(z) = max(\alpha z, z)$, donde el hiperparámetro $\alpha$ define cuanto "filtra" la función, creando una pendiente suave que permite que los gradientes negativos fluyan.

En un paper de 2015 [[49]](https://homl.info/49) se comparan variantes de ReLU y se obtiene un mejor desempeño en todas, sobre la función original. Para Leaky ReLU se suele usar $\alpha = 0.01$.

#### ELU y SELU

En un paper de 2015 [[50]](https://homl.info/50) se plantea una función de activación que superaba el desempeño de todas las variantes de ReLU, reduciendo tiempo de entrenamiento y mejorando las predicciones de test.

$$
\text{ELU}_{\alpha}(z) = 
\begin{cases} 
\alpha(\exp(z) - 1) & \text{if } z < 0 \\ 
z & \text{if } z \geq 0 
\end{cases}
$$

Aunque el resultado de esta función sea parecido al de la ReLU, tiene varios puntos en los que se diferencian:

- Devuelve valores negativos para $z<0$, lo cual alivia el problema del desvanecimiento de gradientes. El hiperparámetro $\alpha$ define el valor negativo que devuelve la función cuando $z$ es negativo.
- Evita el problema de neuronas muertas, ya que tiene un valor $\neq 0$ para $z<0$
- Si $\alpha = 1$, la función es suave en cualquier punto, lo cual ayuda a acelerar el cálculo del descenso de gradiente

Para utilizar ELU en PyTorch solo hay que utilizar el módulo `nn.ELU` junto a la inicialización Kaiming.

En 2017 se planteó una variación de ELU, **SELU** (scaled ELU) en un paper [[selu]](https://homl.info/selu). En dicho paper se demostró que si se implementa una red neuronal de capas densas con SELU, la red se auto-normaliza automáticamente.

Para utilizarlo se ha de llamar al módulo `nn.SELU` con las siguientes restricciones:

- Las entradas deben estar normalizadas, con una media de 0 y desviación estándar 1
- Todos los pesos de las capas ocultas tienen que estar inicializadas con el método de LeCun
- La auto normalización no está garantizada con arquitecturas que no sean MLPs llanas
- No se pueden utilizar técnicas de regularización como ℓ1 o ℓ2, normalización de lotes, capas, max o dropout

#### GELU, Swish, SwiGLU, Mish y ReLU²

La función **GELU** (Gaussian Error Linear Unit) fue presentada en un paper de 2016 [[gelu]](https://homl.info/gelu) y se puede considerar una variante suave de ReLU. Se define con la siguiente fórmula:

$$
\large
GELU(z) = z\Phi(z)
$$

Siendo $\Phi$ una función de distribución acumulativa Gaussiana (CDF): $\Phi(z)$, que corresponde a la probabilidad de que un valor muestreado al azar de una distribución normal con media 0 y varianza 1 sea menor que $z$.

**Swish** nace como una variación de GELU en un paper de 2017 [[swish]](https://homl.info/swish), una aproximación de la función de activación para aligerar el tiempo de computación de los gradientes:

$$
\large
Swish(z)= z\sigma(z)
$$

En el mismo paper se presenta la generalización de Swish añadiendo un parámetro $\beta$ para escalar la entrada de la función de activación:

$$
\large
Swish_{\beta}(z)= z\sigma(\beta z)
$$

Este parámetro puede ser entrenable, normalmente suele tener un único parámetro para el modelo entero o, como mucho, uno por cada capa para mantener el modelo eficiente y evitar el overfitting.

Una variante popular de Swish es **SwiGLU** [[swiglu]](https://homl.info/swiglu), en la cual las entradas pasan por la función de activación Swish y en paralelo por una capa lineal para, finalmente, multiplicarse elemento a elemento:

$$
\large
SwiGLU(z) = Swish_{\beta}(z) ⊗ Linear(z)
$$

Otra función de activación parecida a GELU es **Mish**, planteada en 2019 [[mish]](https://homl.info/mish). Se define de la siguiente manera:

$$
\large
mish(z) = z\tanh(softplus(z))
$$

donde

$$
\large
softplus(z) = \log(1 + \exp(z))
$$

Así como GELU y Swish, es una variante de ReLU suave, no convexa y no monótona.

---

### Normalización de lotes

Si bien la inicialización de Kaiming conjuntamente con ReLU evita los problemas anteriores al principio del entrenamiento, eso no quiere decir que no vaya a volver a lo largo del proceso.

Dado lo anterior, en un paper de 2015 [[51]](https://homl.info/51) se plantea la **normalización de lotes** (Batch Normalization, BN), la cual consiste en añadir una operación antes o después del cálculo de la función de activación para normalizar las entradas de cada capa.

Con el objetivo de hacer lo anterior, el algoritmo necesita aproximar la media y la desviación estándar de cada entrada, evaluando estas variables sobre el lote actual, de la siguiente manera:

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
- $\mu_B$: el vector con la media de las entradas sobre el lote $B$
- $m_B$: el número de instancias en el lote
- $x^{(i)}$: el vector de entrada del lote para la instancia $i$
- $\sigma_B^2$: el vector con las desviaciones sobre el lote $B$
- $\widehat{\mathbf{x}}^{(i)}$: el vector de entradas normalizado y centrado en $0$ para la instancia $i$
- $\varepsilon$: término suavizante, para evitar divisiones por $0$ y que los gradientes no exploten
- $\gamma$: el vector del parámetro de escalado de la salida para la capa
- $⊗$: representa la multiplicación elemento a elemento
- $\beta$: el movimiento del vector de parámetros de salida
- $\mathbf{z}^{(i)}$: la salida de la operación BN (la versión de la entrada rescalada y movida)

En definitiva, hay 4 vectores de parámetros aprendidos en cada capa BN:
- **Aprendidos por backpropagation:**
  - $\gamma$
  - $\beta$
- **Aproximados utilizando media exponencial móvil:**
  - $\mu$
  - $\sigma^2$

Es necesario resaltar que los dos últimos vectores de parámetros se aprenden en el entrenamiento, pero solo son usados una vez este acaba. En el momento en el que utilizamos `model.eval()`, estos parámetros son fijos.

Estos avances permitieron grandes mejoras en los modelos de clasificación de imágenes. Se evitó en gran medida el desvanecimiento de gradientes, hasta el punto de poder utilizar redes muchísimo más profundas sin que colapsaran.

Esta estrategia añade complejidad al modelo, volviéndolo más lento en la inferencia. Sin embargo, es posible fusionar las capas BN con las capas previas después de entrenar, pudiendo evitar esta penalización.

---

### Normalización de capas

La **normalización de capas** (Layer Normalization, LN) es parecida a la de lotes, pero en vez de normalizar a lo largo de la dimensión del lote, lo hace en la dimensión de las características. Así se presentó en un paper de 2016 [[ln]](https://homl.info/ln).

Una de las ventajas que tiene LN es que puede calcular en el momento las estadísticas independientemente de cada instancia, con lo que se comporta igual en entrenamiento como en test. Este método es especialmente útil en arquitecturas como Transformers.

---

### Recorte de gradiente

El **recorte de gradiente** (Gradient Clipping) es una técnica que se utiliza para evitar la explosión de gradientes. Se basa en definir un umbral que no pueden sobrepasar en el momento de backpropagation. Esta técnica se suele usar en redes recurrentes y en algunos casos especiales.

---

## Reusar capas preentrenadas

Normalmente no es la mejor idea entrenar una red profunda desde cero sin buscar otra que ya intente resolver el mismo problema. En el caso de encontrarla, se pueden reusar la mayoría de sus capas, especialmente las inferiores, que suelen aprender características generales.

En el caso de tener acceso a una red profunda entrenada para clasificar imágenes en varias clases, incluidas imágenes de coches, y tu idea es implementar una red que clasifique diferentes tipos de coches, puedes reusar todas las capas excepto la última capa (la que clasifica), que habrá que entrenar desde cero.

> **Nota:** Si la entrada del nuevo modelo no es del mismo tamaño, será necesario aplicar un paso previo de preprocesado para reescalar ese tamaño. Por ejemplo, si la red neuronal preentrenada espera imágenes de 224×224 pero tu nuevo dataset tiene imágenes de 128×128, habrá que reescalarlas.

El objetivo es encontrar el número correcto de capas a reusar. Se puede empezar "congelando" (`requires_grad=False`) las capas para no modificarlas y entrenar el modelo para ver su desempeño. Si el modelo funciona bien, se pueden descongelar las últimas capas y entrenarlas también.

---

### Preentrenamiento no supervisado

En el caso de tener una tarea compleja, no disponer de un modelo preentrenado y no tener mucha información etiquetada ni poder obtener más, existe la posibilidad de aplicar el **preentrenamiento no supervisado**.

Actualmente se suele entrenar el modelo completo sobre la información no etiquetada de una sola tirada, en vez de ir entrenando capa por capa congelando las anteriores para entrenar nuevas. Además, esta estrategia se ha vuelto menos relevante con el surgimiento de los Transformers y modelos de lenguaje grandes.

---

### Preentrenamiento en una tarea auxiliar

Una última opción es entrenar una primera red neuronal en una **tarea auxiliar**, de la cual se puede obtener o generar fácilmente información etiquetada, para después reusar las capas inferiores en la tarea real.

**Ejemplo:**
> Se quiere implementar un sistema de reconocimiento de rostros, únicamente con 2 o 3 imágenes por individuo. Ya que obtener cientos de imágenes por persona no es plausible, se podría utilizar un dataset público con muchas imágenes de rostros para preentrenar el modelo, y luego fine-tuning con el dataset pequeño de la tarea real.

---

## Optimizadores más rápidos

Hasta ahora hemos visto varios métodos para acelerar el entrenamiento de las redes neuronales profundas:
- Estrategias de inicialización
- Funciones de activación
- Normalización de lotes y capas
- Reutilización de capas

Ahora veremos varios optimizadores más allá del optimizador de descenso de gradiente normal.

---

### Momentum

La idea de la **optimización del momentum** presentada en un paper de 1964 [[54]](https://homl.info/54) se basa en afectar el gradiente calculado con una variable dentro del vector de momentum $m$ multiplicada por un factor $\beta$ (típicamente 0.9):

$$
\large
\theta \leftarrow \theta - \eta \nabla_{\theta}J(\theta)
$$

En otras palabras, en la optimización del momentum, el gradiente es usado como una fuerza de aceleración, no como una velocidad. Para evitar que aumente indefinidamente, se implementa un factor de fricción:

$$
\large
\begin{align}
1. \quad & m \leftarrow \beta m - \eta \nabla_{\theta}J(\theta) \\ 
2. \quad & \theta \leftarrow \theta + m 
\end{align}
$$

Ya que en la práctica los gradientes no son constantes, puede no apreciarse tanto la aceleración del entrenamiento, pero esta optimización escapa mucho antes de las mesetas que el descenso de gradiente estándar.

---

### Gradiente acelerado de Nesterov

Yurii Nesterov propuso en su paper de 1983 [[55]](https://homl.info/55) una variante a la optimización de momentum, el **gradiente acelerado de Nesterov** (NAG, Nesterov Accelerated Gradient), la cual mide el gradiente de la función de pérdida evaluada un paso adelante en la dirección del momentum:

$$
\large
\begin{align}
1. & \quad m \leftarrow \beta m - \eta \nabla_{\theta}J(\theta+\beta m) \\ 
2. & \quad \theta \leftarrow \theta + m 
\end{align}
$$

Esta modificación funciona ya que normalmente el gradiente apuntará en la dirección correcta, con lo que será aún más correcto usar el gradiente medido más allá en esa dirección, en vez del gradiente en la posición actual.

---

### AdaGrad

Considerando el problema del "cuenco alargado" en el que el descenso de gradiente comienza yendo rápidamente por la pendiente más acentuada, aunque no sea la que conduzca al mínimo global, para después ralentizarse al acercarse, el algoritmo **AdaGrad** intenta adaptarse a la geometría de la función de pérdida:

$$
\large
\begin{align}
1. & \quad s \leftarrow s + \nabla_{\theta}J(\theta) \otimes \nabla_{\theta}J(\theta)\\ 
2. & \quad \theta \leftarrow \theta - \eta \nabla_{\theta}J(\theta) \oslash \sqrt{s + \epsilon}
\end{align}
$$

> El símbolo $⊗$ representa multiplicación elemento por elemento, mientras que $⊘$ representa división elemento por elemento

El primer paso del algoritmo acumula el cuadrado de los gradientes en el vector s, lo cual es equivalente a calcular $s_i \leftarrow s_i + (\frac{\partial J(\theta)}{\partial \theta_i})^2$ para cada parámetro $i$.

El segundo paso es casi idéntico al descenso de gradiente, con una diferencia notable: el vector de gradientes es escalado por un factor $\sqrt{s + \epsilon}$, siendo $\epsilon$ un factor pequeño para evitar divisiones por cero.

Este algoritmo hace decaer el learning rate, pero lo hace mucho más rápido para dimensiones empinadas que para cuestas suaves, lo que se conoce como **aprendizaje adaptativo**.

Sin embargo, si bien funciona correctamente para problemas cuadráticos, puede parar muy pronto el entrenamiento debido a que escala tanto el learning rate que es incapaz de llegar al punto óptimo global.

---

### RMSProp

**RMSProp** es una variante de AdaGrad que evita que el entrenamiento pare antes de converger por la desaparición del learning rate, acumulando solo los gradientes de las iteraciones más recientes, en vez de todos los históricos:

$$
\large
\begin{align}
1. & \quad  s \leftarrow \alpha s + (1 - \alpha) \nabla_{\theta}J(\theta) \otimes \nabla_{\theta}J(\theta)\\ 
2. & \quad \theta \leftarrow \theta - \eta \nabla_{\theta}J(\theta) \oslash \sqrt{s + \epsilon}
\end{align}
$$

> Siendo $\alpha$ el hiperparámetro del ratio de desintegración, normalmente definido a 0.9

---

### Adam

**Adam** (Adaptive Moment Estimation) combina las ideas de la optimización del momentum y el algoritmo RMSProp. Este optimizador tiene en cuenta la media móvil exponencial de los gradientes y la media móvil exponencial de los cuadrados de los gradientes:

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

> Siendo $t$ el número de la iteración, $\beta_1$ sería el momentum y $\beta_2$ correspondría al ratio de desintegración $\alpha$ de RMSProp

Mientras que los pasos 1, 2 y 5 son muy similares a los de Momentum y RMSProp, los pasos 3 y 4 son correcciones de sesgo para los valores de $m$ y $s$. Ya que al inicializarse a $0$ estarían sesgados estrepitosamente en las primeras iteraciones, estas correcciones los dessesan.

#### AdaMax

Adam acumula los cuadrados de los gradientes en $s$, como hemos visto antes, además de escalar por debajo las actualizaciones de los parámetros por la raíz de $s$. Es decir, Adam escala las actualizaciones por la norma $l_2$ de los gradientes históricos.

Por otro lado, **AdaMax**, introducido en el mismo paper que Adam, reemplaza $l_2$ por $l_{\infty}$. En concreto, reemplaza el paso 2 del algoritmo con:

$$
\large
s \leftarrow \max(\beta_2 s, |\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})|)
$$

elimina el paso 4 y en el paso 5 escala la actualización de gradientes por $s$; el máximo del valor absoluto de los gradientes desintegrados en el tiempo.

En definitiva, esta variante puede hacer que el algoritmo sea más estable que Adam, pero depende bastante del conjunto de datos y en general Adam tiene un mejor desempeño.

#### NAdam

Esta variante es la optimización de Adam con el truco de Nesterov añadido, con lo que llegará a la convergencia más rápido que el algoritmo de Adam base.

#### AdamW

**AdamW** es otra variante de Adam que integra la técnica de regularización de "desintegración de pesos" (weight decay). Esa reduce el tamaño de los pesos de los modelos en cada iteración multiplicándolos por un factor ligeramente menor que 1.

---

### Conclusión

Todas las técnicas que se han visto dependen de las Jacobianas, que miden la pendiente de la función de pérdida. Aunque hay más algoritmos de optimización basados en las Hessianas, midiendo la curvatura de la función de pérdida, estos suelen ser más complejos y costosos computacionalmente.

Para acabar, se resume en la siguiente tabla todas las técnicas de optimización con una puntuación del 1-3 en su velocidad de convergencia y la calidad de esta:

| Optimizador | Velocidad de convergencia (1-3) | Calidad de convergencia (1-3) |
|---|---|---|
| SGD | 1 | 3 |
| SGD (momentum=...) | 2 | 3 |
| SGD (momentum=..., nesterov=True) | 2 | 3 |
| AdaGrad | 2 | 1 *(stops too early)* |
| RMSprop | 2 | 2 o 3 |
| Adam | 2 | 2 o 3 |
| AdaMax | 2 | 2 o 3 |
| NAdam | 2 | 2 o 3 |
| AdamW | 2 | 2 o 3 |

---

## Programación de la tasa de aprendizaje

Encontrar una buena tasa de aprendizaje es importantísimo. Si es muy baja el entrenamiento del modelo será muy lento e incluso puede quedarse atascado en un punto óptimo local. Si, por otra parte, es muy alta, el modelo puede divergir y nunca converger.

---

### Programación exponencial

En este método se aplica un factor $\gamma$ que multiplica el learning rate en intervalos regulares, con lo que después de $n$ épocas, la tasa de aprendizaje será igual al valor inicial multiplicado por $\gamma^n$. Típicamente se usa $\gamma = 0.1$.

---

### Recocido cosinusoidal

En vez de disminuir la tasa de aprendizaje de manera exponencial, se puede utilizar la función de coseno para ir desde la tasa máxima $\large\eta_{max}$ hasta el valor mínimo $\large\eta_{min}$ en cada ciclo:

$$
\large
\eta_t = \eta_{min} + \frac{1}{2} (\eta_{max} - \eta_{min}) ( 1 + \cos(\frac{t}{T_{max}}\pi))
$$

Ya que es difícil definir tanto el número total de épocas como la tasa mínima, es preferible utilizar otros métodos como el siguiente.

---

### Programador por desempeño

También conocido como **programación adaptativa**, sigue una métrica en concreto durante el entrenamiento y, si esta métrica deja de mejorar por un tiempo, multiplica el learning rate por un factor predefinido.

En PyTorch existe `ReduceLROnPlateau` que implementa esta estrategia con los siguientes parámetros:

- **mode**: define si la métrica a seguir tiene que maximizarse (`max`) o minimizarse (`min`). Es decir, en `max` se reducirá la tasa si la métrica ha dejado de aumentar y en `min` se reducirá si la métrica ha dejado de disminuir.
- **patience**: define el número de iteraciones que espera para ver una mejora en la métrica a seguir antes de reducir el learning rate.
- **factor**: define el valor por el que se reducirá el learning rate.

---

### Calentar la tasa de aprendizaje

Los métodos anteriores empiezan todos con un valor de $\large\eta$ máximo, lo que puede llevar a que exploten los gradientes o que nunca se haga un avance significativo en algunos casos. Por eso, es común "calentar" la tasa de aprendizaje al principio del entrenamiento.

Una implementación de esta idea es un programador lineal que aumente la tasa de forma lineal durante varias épocas, por ejemplo del 10-100% durante 3 iteraciones.

En resumidas cuentas, normalmente siempre quieres "calentar" la tasa de aprendizaje al principio del entrenamiento, así como también quieres "enfriarla" cuando está acabando de entrenar.

También hay otros casos, como cuando el descenso de gradiente se queda atascado en un valle o un punto óptimo local donde pueda quedarse durante todo el entrenamiento. En estos casos, necesitaremos modificar el learning rate para escapar:

- Modificar a mano durante el entrenamiento el valor del learning rate
- Implementar una solución personalizada que tenga en cuenta una métrica (parecido a `ReduceLROnPlateau`)
- Utilizar Recocido cosinusoidal con reinicios calientes

---

### Recocido cosinusoidal con reinicios calientes

Este método presentado en un paper de 2016 [[coslr]](https://homl.info/coslr) repite en bucle el funcionamiento del método original. Se recalcula utilizando la función de coseno el valor del learning rate constantemente, reiniciando a un valor alto periódicamente para escapar de mínimos locales.

---

### Programación de un ciclo

Otro método popular, introducido en un paper en 2018 [[1cycle]](https://homl.info/1cycle), comienza calentando el learning rate con un valor $\eta_0$ que crece linealmente hasta $\eta_1$ a mitad del entrenamiento, para después disminuir linealmente hasta un valor $\eta_2$ mucho más bajo al final del entrenamiento.

---

## Evitar overfitting mediante la regularización

(Contenido próximo...)
