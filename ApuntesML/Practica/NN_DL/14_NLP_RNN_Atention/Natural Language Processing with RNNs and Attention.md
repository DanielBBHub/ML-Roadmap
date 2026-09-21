# Natural Language Processing with RNNs and Attention

## Generando textos shakespiresco utilizando una RNN de caracteres

### Creando el conjunto de datos

#### Tokenización a nivel de carácter

Las redes neuronales solo pueden trabajar con números, así que el primer paso para procesar texto es convertirlo en una secuencia numérica. A este proceso se le llama **tokenización**: consiste en trocear el texto en unidades más pequeñas (**tokens**) y asignar a cada token distinto un identificador entero único.

Los tokens pueden ser de distinto tipo según el nivel de granularidad elegido:

- **A nivel de palabra**: cada palabra del vocabulario es un token.
- **A nivel de carácter**: cada carácter distinto que aparece en el texto es un token.
- **A nivel de subpalabra** (como *Byte Pair Encoding*): un punto intermedio muy usado en la práctica.

En este caso se utiliza una **tokenización a nivel de carácter**, la más sencilla de implementar. El **vocabulario** se construye simplemente recopilando todos los caracteres distintos que aparecen en el corpus, de modo que su tamaño depende del alfabeto y los símbolos usados, no del número de palabras distintas.

A partir del vocabulario se construyen dos diccionarios complementarios:

- `char_to_id`: mapea cada carácter a su identificador entero (necesario para **codificar** texto).
- `id_to_char`: mapea cada identificador de vuelta a su carácter (necesario para **decodificar**, es decir, convertir las predicciones del modelo de nuevo en texto legible).

Con estas tablas de equivalencia, codificar un texto consiste en sustituir cada carácter por su ID y empaquetar el resultado en un tensor; decodificar es el proceso inverso.

#### Modelado de lenguaje como predicción de la siguiente ficha

Para entrenar una RNN a generar texto se plantea el problema como una tarea de **modelado de lenguaje autorregresivo**: dado un fragmento de texto, el modelo debe aprender a predecir cuál será el siguiente carácter.

Para ello, el texto codificado (ya convertido en una única secuencia larga de IDs) se recorre con una **ventana deslizante** de longitud fija (`window_length`). Por cada posición de la ventana se generan dos secuencias:

- La **entrada** (`window`): los **n** caracteres a partir de esa posición.
- El **objetivo** (`target`): la misma ventana desplazada una posición hacia adelante.

De esta forma, el objetivo en cada posición de la secuencia es simplemente el carácter que sigue al correspondiente carácter de entrada. Esto permite que, durante el entrenamiento, el modelo aprenda simultáneamente a predecir "el siguiente carácter" para cada una de las posiciones de la ventana, y no solo para la última, aprovechando mucho mejor cada muestra.

#### Limitaciones de representar los tokens como IDs enteros

Aunque ya se dispone de conjuntos de entrenamiento, validación y test numéricos, usar directamente los IDs enteros como entrada de la red sería un error: al ser simplemente índices arbitrarios del vocabulario, su valor numérico no guarda ninguna relación de similitud entre caracteres, y la red podría interpretar erróneamente que existe una relación de orden o magnitud entre tokens que en realidad no tiene ningún significado.

Una alternativa clásica es el **one-hot encoding**, que representa cada token como un vector binario donde todas las posiciones son cero salvo la correspondiente al token, que vale uno. Esto garantiza que todos los tokens quedan igual de "separados" entre sí, sin introducir relaciones de orden falsas. Sin embargo, esta representación escala muy mal con el tamaño del vocabulario: cada vector tiene tantas dimensiones como tokens distintos existan, lo cual resulta manejable para un vocabulario de caracteres, pero se vuelve inviable en cuanto se trabaja con vocabularios de palabras (que pueden tener decenas o cientos de miles de entradas).

La alternativa que ofrecen las redes neuronales, y que se explora en la siguiente sección, son los **embeddings**: representaciones vectoriales densas y de dimensión reducida que la propia red aprende durante el entrenamiento, y que sí son capaces de capturar relaciones semánticas entre tokens.

### Embeddings

El "embedding" o la "incrustación" es una representación densa de información con varias dimensiones, normalmente una característica categórica. Si hay 50000 posibles características, entonces el "one-hot encodign" porduce un vector disperso de 50000 dimensiones, mientras que el embeding produce un vector relativamente pequeño y denso de, por ejemplo, 3000 dimensiones. El tamaño de estas incrustaciones es un parámetro que se puede configurar

Normalmente estos "embeddings" se inicializan de manera aleatoría y se van entrenando a medida que se hace el descenso de gradiente. Ya que estos son entrenables e iran mejorando, esto hará que se vayan acercando unos a otros, ya que representan categorías relativmante parecidas, con lo que cuanto mejor sea la representación de las características, más fácil será para las redes neuronales hacer predicciones más certeras. Esto se conoce como "representation learning".

Y no solo servirán como buenas representaciones de las características para la tarea en la que se esta aplicando si no que normalmente estos podrán reusarse; si vas a implementar una solución será mejor utilizar un embedding preentrenado, como con los modelos, que entrenar uno de cero.

Dependiendo del entrenamiento de las incrustaciones representara de una u otra forma la relación entre las palabras del texto procesado, haciendo que las palabras como los sinonimos o con relaciones semanticas tengan una distancia pequeña entre ellas (como España, Francia e Italia). Pero no todo es la distancia, las incrustaciones de palabras también estan ordenadas en los ejes por importancia; si calculaesmos Rey - Hombre + Mujer el resultado sería Reina o España - Madrid + Francia sería París, con lo que puede llegar a codificar la relación de género o la noción de capital.

Pero no todo es positivo, dependiendo del conjunto de datos, estas incrustaciones pueden capturar los sesgos del autor, ya que si bien puede reconocer que El hombre es a "rey" lo mismo que la mujer es a "reina, de la misma manera "aprenderá" que El hombre es a "doctor" lo que la mujer es a "enfermera".

### Generando texto nuevo al estilo Shakespeare

Una vez entrenado el modelo, generar texto nuevo consiste en aplicar una y otra vez la misma idea de **modelado autorregresivo** que se usó para entrenarlo: se le da un texto inicial, se le pide que prediga el siguiente carácter, ese carácter se añade al final del texto, y el texto ampliado se vuelve a pasar por el modelo para predecir el siguiente, y así sucesivamente.

#### Decodificación voraz frente a muestreo

La forma más directa de elegir "el siguiente carácter" es quedarse siempre con el más probable según el modelo (el `argmax` de las probabilidades de salida). A esto se le llama **decodificación voraz** (*greedy decoding*). El problema es que en la práctica tiende a quedarse atascada repitiendo las mismas palabras o frases una y otra vez, porque en cuanto el modelo entra en un patrón de alta probabilidad, siempre va a elegir continuarlo.

La alternativa es **muestrear** el siguiente carácter en vez de escoger siempre el más probable: si el modelo estima una probabilidad *p* para un token dado, ese token se elige con probabilidad *p*, no con certeza. Esto permite que aparezcan de vez en cuando caracteres que no son los más probables, generando un texto más variado e interesante. En PyTorch, esto se hace con `torch.multinomial()`, que dada una lista de probabilidades por clase, devuelve índices de clase muestreados según esas probabilidades (no necesariamente el de mayor probabilidad).

#### La temperatura como control de la diversidad

Para poder ajustar cuánto se parece este muestreo a la decodificación voraz (siempre lo más probable) o a un muestreo totalmente uniforme (todos los caracteres igual de probables), se introduce la **temperatura**: un número por el que se dividen los logits antes de aplicar la función `softmax` para obtener las probabilidades.

- Una temperatura **cercana a cero** favorece mucho a los caracteres de mayor probabilidad, acercando el comportamiento a la decodificación voraz (y por tanto, a las repeticiones).
- Una temperatura **alta** aplana la distribución de probabilidad, dando a todos los caracteres una probabilidad más parecida entre sí, lo que genera texto más caótico y menos coherente.

Temperaturas bajas suelen preferirse para generar texto que debe ser rígido y preciso (por ejemplo, ecuaciones), mientras que temperaturas moderadas son mejores para generar texto creativo y variado sin perder coherencia. Si la temperatura es demasiado alta, el texto generado deja de parecerse al lenguaje real.

#### Generando el texto carácter a carácter

Esta lógica se puede encapsular en dos funciones que trabajan juntas: una que decide un único carácter nuevo dado el texto actual (aplicando softmax con temperatura sobre los logits del último paso de la secuencia y muestreando con `torch.multinomial`), y otra que llama a la primera repetidamente, añadiendo cada carácter nuevo al texto antes de pedir el siguiente. Esta segunda función es la que implementa directamente la idea de generación autorregresiva: el texto de salida de cada paso es parte de la entrada del paso siguiente.

#### Limitaciones y alternativas más avanzadas

El modelo solo puede aprender a reproducir patrones que quepan dentro de `window_length` (aquí, 50 caracteres): no es capaz de mantener coherencia con algo que apareció mucho antes de esa ventana. Ampliar la ventana ayuda, pero también complica el entrenamiento, y ni siquiera las LSTM o GRU manejan bien secuencias muy largas.

El muestreo con temperatura no es la única técnica de generación. Otras alternativas habituales son:

- **Top-k sampling**: muestrear solo entre los *k* caracteres más probables, descartando el resto.
- **Top-p (nucleus) sampling**: muestrear solo entre el conjunto más pequeño de caracteres cuya probabilidad acumulada supera un umbral dado.
- **Beam search**: en vez de generar un único texto carácter a carácter, mantener varias continuaciones candidatas en paralelo y quedarse con la más probable en conjunto (se retoma más adelante en el capítulo).

También ayuda, simplemente, usar más capas o neuronas en la GRU, entrenar durante más tiempo, o añadir más regularización.

#### Una curiosidad: la neurona del sentimiento

Aunque el char-RNN solo se entrena para predecir el siguiente carácter, esta tarea aparentemente simple obliga al modelo a aprender también estructura de más alto nivel: para acertar el carácter que sigue a "Great movie, I really " ayuda mucho entender que la frase es positiva, y que por tanto es más probable que continúe con "l" (de *loved*) que con "h" (de *hated*).

De hecho, un estudio de OpenAI de 2017 (Radford et al.) entrenó un modelo similar a gran escala y descubrió que una de sus neuronas actuaba, por sí sola, como un clasificador de sentimiento con resultados comparables a los mejores modelos supervisados de la época — sin haber visto ni una sola etiqueta de sentimiento durante el entrenamiento. Este hallazgo fue una de las motivaciones que impulsaron el preentrenamiento no supervisado en NLP, la idea que dio pie a modelos como los que se explorarán más adelante en este mismo capítulo.