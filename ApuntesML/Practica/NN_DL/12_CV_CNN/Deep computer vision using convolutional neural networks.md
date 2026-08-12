# Deep computer vision using convolutional neural networks

## Capas convolucionales

La parte mas importante de una red convolucional son las capas convolucionales: las neuronas en la primera capa convolucional no se conectana cada pixel en la imagen de entrada, solo a pixeles en sus campos receptivos, a su vez, las neuronas de la siguiente capa se centran en un rectangulo ubicado dentro del campo receptivo de la capa anterior.

Esta arquitectura permite a la neurona concentrarse en las características de "bajo nivel" en las primeras capas intermedias para luego ensamblarlas dentro de características de más "alto nivel" en las siguientes capas. Esta estructura jerárquica es idónea para manejar los objetos compuestos que forman las imágenes del mundo real.

Una neurona en la posicion $(i,j)$ de una capa esta conectada a la salida de las neuronas en la capa anterior ubicadas en $(i \rightarrow i + f_n, j \rightarrow j + f_w)$ donde $f_n$ y $f_w$ son el alto y ancho del campo receptivo. Para que una capa intermedia tenga la misma altura y anchura que la capa anterior se suele utilizar el padding de ceros (añadir ceros alrededor del input).

También es posible conectar una capa de entrada grande a una mucho mas pequeña separando los campos receptivos, reduciendo dramaticamente la complejidad computacional del modelo (El tamaño del paso horizontal/ vertical de un campo receptivo a otro se llama "stride"). Una capa de entrada $5 \times 7$ con padding de ceros puede conectarse a otra capa $3 \times 4$ utilizando campos receptivos de $3 \times 3$ y un "stride" de 2 en ambas direcciones. En general, una neurona en $(i,j)$ de la capa superior es conectada a las salidas de las neuronas de la capa anterior en las filas $i \times s_h \rightarrow i \times s_h+f_h-1$ y las columnas $j \times s_w \rightarrow i \times s_w+f_w-1$ donde $s_h$ y $s_w$ son los "strides" verticales y horizontales

### Filtros

Los pesos de las neuronas se pueden representar como una imagen pequeña del tamaño del campo receptivo. Estos filtros (o kernels) son areas superpuestas a las entradas que escalan los valores. Un ejemplo son los filtros verticales u horizontales, los cuales son, por ejemplo, matrices $7 \times 7$ con todos los valores a 0 menor las columnas o filas centrales, lo cual fuerza a la neurona a únicamente fijarse en esas regiones.

> En el caso de utilizar como entrada una imágen y emplear un filtro vertical, el resultado de la red seria un "mapa de características" en el que la imágen se vería emborronada como si el objeto se hubiese movido verticalmente en el momento de la toma de la imágen. Esto mismo pasaría con el filtro horizontal. Lo que está pasando es que están "resaltando" las líneas verticales/horizontales de la imagen

### Combinar diferentes mapas de características

Las capas convolucionales pueden tener varios filtros, lo cual resulta en un mapa de características por cada uno de los filtros. La salida de estas operaciones se representa de mejor manera en un diagrama 3D.

Hay una neurona por cada pixel de cada mapa de característias y todas las neuronas dentro de un mapa de características comparten los mismos parámetros (kernel, bias, ...), así como las neuronas en mapas de características diferentes utilizan diferentes parámetros. El campo receptivo de una neurona es el mismo, pero se extiende a todos los mapas de características de la capa. En resumen, una capa convolucional aplica simultáneamente filtros entrenables a sus entradas siendo capaz de detectar varias características en cualquier punto de su entrada.

Adicionalmente, las imágenes estan compuestas por subcapas:

- Grayscale: 1 capa de intensidad 0-255
- Color: 3 capas con intensidad 0-255 para formar colores
- Satelite: 3 capas con intensidad 0-255 para formar colores + diferentes frecuencias de la luz

En especifico, una neurona en $(i,j)$ en el mapa de características $k$ en una capa $l$ esta conectada a la salida de las neuronas de la capa anterior $l-1$ en $i \times s_h \rightarrow i \times s_h+f_h-1$ y $j \times s_w \rightarrow i \times s_w+f_w-1$ a lo largo de todos los mapas de características de la capa $l-1$. Es decir, dentro de una misma capa, todas las neuronas en la misma fila $i$ y columna $j$ pero en diferentes mapas de características estan conectados a las salidas de las mismas neuronas de la capa anterior.

Para calcular la salida de una neurona en una capa convolucional se utiliza la siguiente equación:
$$
\large
z_{i,j,k} = b_k + \sum_{u=0}^{f_h-1} \sum_{v=0}^{f_w-1} \sum_{k'=0}^{f_{n'}-1} x_{i',j,k'} \times w_{u,v,k',k} \quad \text{with} \quad \begin{cases} i' = i \times s_h + u \\ j' = j \times s_w + v \end{cases}
$$

donde :
- $\large z_{i,j,k}$ es la salida de la neurona en $(i,j)$ del mapa $k$ de la capa $l$
- $\large s_h$ y $s_w$ son los "strides" verticales y horizontales
- $\large f_h$ y $f_w$ son la altura y anchura del campo receptivo 
- $\large f_{n'}$ es el número de mapas de características de la capa anterior $l-1$
- $\large x_{i',j,k'}$ es la salida de la neurona de la capa anterior $l-1$, en $(i',j')$ del mapa, o canal si es la capa de entrada$ $k'$
- $\large b_k$ es el parametro bias
- $\large w_{u,v,k',k}$ es el peso de conexión entre cualquier neurona en el mapa $k$ de la capa $l$ y su entrada en $(u,v)$ y mapa $k'$

## Capas de agrupacion (Pooling)
La función esencial de las capas pooling es reducir las imagenes de entrada para disminuir la carga computacional, el uso de memoria y el numero de parámetros.

Así como las capas convolucionales, cada neurona en una capa de agrupación esta conectada a las salidas de un número de neuronas de la capa anterior que se encuentran en un campo receptivo. Se ha de definir el tamaño, el "stride" y el tipo de padding como en las capas convolucionales, pero estas capas no tienen ni pesos ni bias, solo agrega mediante una función (como max o media) las entradas.

> Normalmente estas capas de agrupación se aplican independientemente del canal, con lo que la imágen de entrada y la resultante tienen la misma profundiad (número de canales)

Además de las características anteriores, la capa de pooling max tiene cierta cantidad de invarianza a pequeñas traslaciones en la imagen, con lo que introduciendo una capa de estas cada x capas convolucionales podemos obtener invarianza a gran escala, así como una pequeña tolerancia a rotación y escala de la imagen. Esto puede ayudar al modelo a tomar una predicción en la que no dependa de estas variables.

Por otro lado, estas capas son altamente destructivas, reduciendo el tamaño de la imagen, sus detalles, área ... así como contraproducentes en aplicaciones en las que es importante la traslación/rotación/escalado de los objetos en la imagen.

