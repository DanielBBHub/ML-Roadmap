# Processing Sequences Using RNNs and CNNs

## Capas y neuronas recurrentes

Hasta ahora nos hemos concentrado en las redes que se movian en un único sentido, hacia delante. Estas neuronas recurrentes pueden transmitir información pero tambien tienen conexiones apuntando hacia atras

Si nos fijamos en laa red neuronal recurrente mas simple, la que esta compuesta de una neurona, podemos ver que esta esta recibiendo una entrada, generando una salida y enviando esta salida de vuelta a ella misma. En cada punto en el tiempo $t$, esta neurona recurrente recibe una entrada $x_{(t)}$, así como su propia salida anterior, $y_{(t-1)}$ (si es el primer paso y no hay una salida anterior se suele definir a $0$). En el caso de formar una capa recurrente, cada paso $t$ la capa recibe un vector de entrada $x_{(t)}$, así como un vector de la salida anterior $y_{(t-1)}$

Cada neurona recurrente tiene dos pares de pesos: una para las entradas y otras para las salidas anteriores; $w_x$ y $w_y$ o $W_x$ y $W_y$ cuando tratamos con una capa recurrente (escalares para neuronas, vectores para capas)

El vector de salida de estas capas se define de la siguiente manera:

$$
\large
y_{(t)} = \phi (W_x^t x_{(t)} + W_{\^{y}}^T \^{y}_{(t-1)} + b)
$$

De la misma manera que en las redes anteriores, podemos calcular la salida de la capa para un lote completo poniendo todas las entradas en el punto $t$ en una matriz de entrada $X_{(t)}$

$$
\large
\begin{align}
\^{Y}_{(t)} = \phi(X_{(t)}W_x + \^{Y}_{(t-1)}W_{\^{y}} + b) = \\
= \phi([\begin{matrix} X_{(t)} & \^{Y}_{(t-1)}\end{matrix}]W +b) \text{ siendo } W = \begin{matrix} W_x \\ W_{\^{y}}\end{matrix}
\end{align}
$$

En la ecuación anterior:

- $\^{Y}_{(t)}$ es una matrix $m \times n_{\text{neuronas}}$ que contiene las salidas en $t$ para cada instancia del lote ($m$ es una instancia del lote y $n_{\text{neuronas}}$ es el numero de neuronas)

- $X_{(t)}$ es una matriz $m \times n_{\text{entradas}}$ que contiene las entradas de cada una de las instancias

- $W_x$ es una matriz $n_{\text{entradas}} \times n_{\text{neuronas}}$ que contiene todas las conexiones de pesos para las entradas en cada punto $t$

- $W_{\^{y}}$ es una matriz $n_{\text{neuronas}} \times n_{\text{neuronas}}$ que contiene todas las conexiones de los pesos para las salidas de los pasos anteriores

- $b$ es un vextor de tamaño $n_{\text{neuronas}}$ con los bias de cada una de las neuronas

> Es importante destacar que $\^{Y}_{(t)}$ es una función de $X_{(t)}$ y $\^{Y}_{(t-1)}$, que a su vez es una función de $X_{(t - 1)}$ y $\^{Y}_{(t-1)}$. De esta manera contiene todas las salidas desde el principio del tiempo para $t=0$, donde se asume que todo son $0$ menos la entrada

### Celulas de memoria

Ya que las entradas de una neurona recurrente en un punto $t$ son producto de todas las entradas en puntos $t$ anteriores, podria decirse que ha formado una especie de "memoria".

El estado de una celula en un punto $t$, definida como $h_{(t)}$, es una función de algunas entradas de ese paso y el estado anterior $\large h_{(t)} = f(x_{(t)}, h_{(t-1)})$. La salida $\^{y}_{(t)}$ es tambien una función de los estados anteriores y de la entrada actual y normalmente es únicamente una función dele stado actual.

### Secuencias de salida y entrada

Una RNN puede recibir a la vez una secuencia de entradas y producir una secuencia de salidas. Este tipo de red "secuencia a secuencia" es util para predecir series de tiempo (consumo de energia de casa, velocidades ...)

Por otro lado, las redes "secuencia a vector" son las que ignoran todas las salidas menos la última (predicción de sentimiento de una reseña de una pelicula)

También podría introducirse el mismo vector de entrada una otra vez y que la red sacase una secuencia. Esto es conocido como una red "vector a secuencia" (la entrada podria ser una imagen y la salida una frase sobre ella)

Finalmente, podria montarse una red "secuencia a vector" llamada encoder, seguida por una red "vector a secuencia" llamada decoder (traducción de una oración de un idioma a otro). El encoder podria recibir un secuencia (oración en idioma X) y generar un vector representativo que entraría al decoder, que descodificaria este vector en una secuencia (oración en idioma Y)

## Entrenar RNNs

Para el entrenamiento de RNN, el truco es "desenrollarla" en el tiempo y utilizar backprop normal, conocido como backprop en el tiempo.

De la misma manera que hasta ahora, se pasan los resultados hacia delante en la red "desenrollada" para que se evaluen las salidas con alguna función de pérdida $ℒ(Y_{(0)},Y_{(1)}, ..., Y_{(T)}; \^Y_{(0)}, \^Y_{(1)}), ..., \^Y_{(T)}$ donde $\^Y_{(T)}$ es la predicción en tiempo $T$ y $Y_{(T)}$ es la etiqueta para la muestra en tiempo $T$. Hay que puntualizar que la función de pérdida puede ignorar algunas salidas, como es el caso con las redes "secuencia a vector".

Finalmente, se calculan los gradientes de las salidas pertinentes y se propagan por la red "desenrollada", actualizando los pesos y los bias

## Prediciendo secuencias de tiempo

Como ejemplo de serie de tiempo real vamos a utilizar el número de pasajeros que montan en bus y en tren en Chicago cada día. Una serie de tiempo es información con valores distintos en puntos de tiempo distintos, normalmente a intervalos regulares (en este caso, diario). Como aquí hay dos valores por cada punto de tiempo (bus y rail), se trata de una serie de tiempo **multivariable**. La tarea más habitual sobre este tipo de datos es predecir valores futuros (forecasting), aunque también se pueden usar para rellenar huecos pasados, clasificación o detección de anomalías.

### Predicción naive

Cuando una serie muestra patrones que se repiten con regularidad (**temporadas** o *seasonality*), una forma sencilla de generar una predicción de referencia es reutilizar valores pasados que encajen con el patrón: por ejemplo, predecir el valor de hoy con el valor de hace 7 días, asumiendo un patrón semanal. A esto se le llama **predicción naive**, y aunque no es una técnica de aprendizaje, sirve como baseline con el que comparar modelos más sofisticados: si un modelo no supera a la predicción naive, no está aportando valor real.

Para medir qué tan buena es una predicción naive (o cualquier otra) se pueden usar métricas como:

- **MAE (error medio absoluto)**: la media del valor absoluto de la diferencia entre la predicción y el valor real.
- **MAPE (porcentaje de error medio absoluto)**: el MAE expresado como porcentaje del valor real, lo que facilita comparar el error entre series con escalas distintas (por ejemplo, bus y rail).

### Resampling y media móvil

Para observar patrones de más largo plazo (por ejemplo, temporadas anuales) puede ser útil **remuestrear** (resample) la serie a una granularidad menor, como de diaria a mensual, y calcular una **media móvil** (rolling average, en este caso de 12 meses) sobre esa serie remuestreada. Esto suaviza el ruido de corto plazo y deja ver con más claridad tendencias y ciclos que a nivel diario quedan ocultos.

### Diferenciación y estacionariedad

Otra técnica muy común es la **diferenciación**: en vez de trabajar con los valores originales, se trabaja con la diferencia entre cada valor y el valor de un desfase determinado (por ejemplo, `diff(12)` sobre datos mensuales para eliminar la estacionalidad anual). Diferenciar permite eliminar tanto la temporalidad como las modas/tendencias de la serie (por ejemplo, una caída sostenida de pasajeros en cierto periodo de años), dejando una serie **estacionaria**: una serie cuyas propiedades estadísticas (media, varianza) no cambian con el tiempo.

Trabajar con series estacionarias simplifica mucho el modelado, ya que se elimina la necesidad de que el modelo aprenda también la tendencia y la estacionalidad. Una vez entrenado y validado un modelo sobre la serie diferenciada, es sencillo recuperar las predicciones en la escala original simplemente deshaciendo la diferenciación (sumando de nuevo el valor de referencia que se restó).

### La familia del modelo ARMA

El modelo ARMA (autoregressive moving average) fue desarrollado por Herman Wold en la década de los 30: cálcula predicciones utilizando una suma ponderada de los valores pasados y corrige esta predicción añadiendo una media móvil.

$$
\large
\begin{align}
\^{y}_{(t)} = \sum_{i=1} \alpha_i y_{(t-i)} + \sum_{i=1} \Theta_i \epsilon_{(t-i)} \\
\text{ dada } \epsilon_{(t)} = y_{(t)} - \^{y}_{(t)}
\end{align}
$$

En la ecuación anterior:
- $\^{y}_{(t)}$ es la predicción del modelo para el paso $t$

- $y_{(t)}$ es el valor de la serie temporal en el paso $t$

- La primera suma es la suma ponderada de los últimos $p$ valores en base a los pesos $\alpha_i$. $p$ es un hiperparametro que determina cuantas muestras en el pasado tiene que tener en cuenta el modelo. Esta suma es el componente autoregresivo del modelo

- La segunda suma es una suma ponderada sobre el error de las últimas $q$ predicciones en base a los pesos $\Theta_i$. De igual manera, $q$ es un hiperparametro. Esta suma es la media móvil del modelo

Es importante destacar que este modelo asume que la serie temporal es estacionaria, si no lo es tendremos que procesar esa información. Para esto se pueden aplicar diferenciaciones a la serie, tantas como hagan falta; aplicar $d$ rondas consecutivas de diferenciación calcula una aproximación de la derivada de orden $d$ de la serie temporal, eliminando modas polinómicas de hasta grado $d$. Este hiperparametro $d$ se le conoce como orden de integración.

Esa es la contribución principal del modelo **ARIME** (autoregressive integrated moving average), el cual calcula $d$ rondas de diferenciación para asegurar que la serie sea estacionaria.

El último miembro de la familia **ARMA** es el modelo **SARIMA** (seasonal ARIMA). Este modelo añade un componente temporal para una frecuencia dada, utilizando los hiperparametros $[p,d,q]$ ademas de los nuevos parametros que integra $[P,D,Q]$ para tener en cuenta las modas temporales y $s$ para denotar el periodo temporal.

La librería statsmodels ofrece la clase ARIMA (dentro de statsmodels.tsa.arima.model), que en realidad implementa toda la familia SARIMA: además del hiperparámetro order=(p, d, q) acepta un seasonal_order=(P, D, Q, s) opcional. Si seasonal_order se deja vacío o en (0, 0, 0, 0), el modelo se reduce a un ARIMA normal.

Al preparar la serie es importante fijar explícitamente su frecuencia (por ejemplo asfreq("D") para frecuencia diaria) ya que el modelo necesita conocer el espaciado temporal entre observaciones para poder extrapolar correctamente.

Validación mediante walk-forward (validación progresiva)

Evaluar el modelo con una única predicción puede llevar a conclusiones erróneas por simple suerte o mala suerte. Para obtener una estimación más fiable del rendimiento se emplea una validación walk-forward: se realizan predicciones a un paso (one-step-ahead) para cada día de un periodo de test, reentrenando el modelo desde cero cada vez con todos los datos disponibles hasta ese momento (incluyendo los días de test ya transcurridos). Con las predicciones acumuladas de todo el periodo se calcula una métrica de error, como el MAE (mean absolute error), que resume el desempeño global del modelo.

Este MAE se compara contra una predicción naive (por ejemplo, asumir que el valor de mañana será igual al de hoy, o al de la misma semana pasada) que sirve como referencia mínima: un modelo solo es útil si consigue batir de forma consistente a esta baseline. En el caso de la serie de pasajeros de rail, el SARIMA obtuvo un MAE de ≈32.041, notablemente mejor que el MAE de la predicción naive (≈42.143), lo cual indica que, aunque el modelo no es perfecto, aporta una mejora sustancial de media.

Selección de hiperparámetros

Elegir los valores de $p, d, q, P, D, Q$ y $s$ no es trivial. Existen métodos analíticos basados en las funciones de autocorrelación (ACF) y autocorrelación parcial (PACF) de la serie diferenciada, que permiten inferir qué órdenes autorregresivo y de media móvil son razonables. Sin embargo, el enfoque más simple para empezar es la búsqueda por fuerza bruta (grid search): se entrena y evalúa el modelo (mediante walk-forward validation) para cada combinación de hiperparámetros candidata, y se elige la combinación con menor error.

Como guía general para acotar el espacio de búsqueda:
- $p, q, P, Q$ suelen tomar valores pequeños, típicamente entre 0 y 2, y en ocasiones hasta 5 o 6.
- $d, D$ suelen ser 0 o 1, y rara vez 2.
- $s$ no se busca: se fija según el periodo de la estacionalidad dominante que se observe en los datos (en este caso, $s=7$ por la fuerte estacionalidad semanal del tráfico de pasajeros).

La métrica de error usada en la búsqueda (MAE en este ejemplo) puede sustituirse por cualquier otra que se ajuste mejor al objetivo de negocio del problema.

Como alternativa (o complemento) a la búsqueda por fuerza bruta, existe un método más analítico basado en dos herramientas estadísticas: la función de autocorrelación (ACF) y la función de autocorrelación parcial (PACF).

- La ACF mide, para cada desfase $k$ (lag), la correlación entre la serie y una versión de sí misma desplazada $k$ pasos. Captura tanto la relación directa entre $y_{(t)}$ e $y_{(t-k)}$ como la que llega indirectamente a través de los desfases intermedios ($y_{(t-1)}, y_{(t-2)}, \dots, y_{(t-k+1)}$).
- La PACF mide esa misma correlación entre $y_{(t)}$ e $y_{(t-k)}$, pero eliminando el efecto de los desfases intermedios. Es decir, aísla la correlación que aporta específicamente el lag $k$, una vez descontado lo que ya explican los lags anteriores.

Estas dos funciones se calculan sobre la serie ya hecha estacionaria (tras aplicar la diferenciación necesaria) y se interpretan como una heurística para elegir $p$ y $q$:

- Si la PACF cae abruptamente a cero después del lag $p$ (mientras la ACF decae de forma más gradual), la serie se comporta como un proceso AR(p) puro.
- Si la ACF cae abruptamente a cero después del lag $q$ (mientras la PACF decae de forma más gradual), la serie se comporta como un proceso MA(q) puro.
- Si ambas decaen gradualmente sin un corte claro, es probable que se necesite un modelo ARMA mixto, y suele ser más práctico recurrir a la búsqueda por fuerza bruta.

El mismo razonamiento se puede aplicar a los desfases estacionales (múltiplos de $s$, por ejemplo 7, 14, 21 para una estacionalidad semanal) para orientar la elección de $P$ y $Q$.

Este método tiene la ventaja de requerir mucho menos cómputo que un grid search exhaustivo, pero exige interpretar gráficamente los correlogramas de ACF y PACF, lo cual puede ser subjetivo cuando la señal no es clara — de ahí que, en la práctica, muchas veces se combine con (o se sustituya directamente por) la búsqueda por fuerza bruta.
