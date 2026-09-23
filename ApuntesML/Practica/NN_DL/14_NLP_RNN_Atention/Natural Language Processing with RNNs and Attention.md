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

## Análisis de sentimientos de texto

### Clasificación de texto: el dataset IMDb

El análisis de sentimientos es una de las aplicaciones más habituales de la clasificación de texto en PLN: dado un fragmento de texto, el modelo debe decidir a qué categoría pertenece (en este caso, si una reseña expresa una opinión positiva o negativa sobre una película). El conjunto de datos de referencia para esta tarea es **IMDb**, formado por reseñas de cine etiquetadas como positivas o negativas; su papel en PLN es equivalente al de MNIST en clasificación de imágenes: un problema sencillo y muy estudiado que sirve de primer banco de pruebas antes de abordar tareas más complejas.

### Tokenización a nivel de subpalabra

La tokenización a nivel de carácter usada para generar texto (ver la sección anterior) no es la única opción, ni siempre la más adecuada. Para tareas de clasificación sobre lenguaje natural con vocabularios grandes es habitual tokenizar a un nivel intermedio entre el carácter y la palabra: la **tokenización a nivel de subpalabra**. La motivación, explorada en un paper de 2016, es resolver el problema de las palabras desconocidas (*out-of-vocabulary*): un modelo que solo conoce palabras completas no sabe qué hacer ante una palabra que no vio en el entrenamiento, mientras que uno capaz de descomponerla en fragmentos más pequeños y conocidos (por ejemplo, "grandísimo" → "grand" + "ísimo") puede seguir infiriendo algo de su significado a partir de esas partes.

#### El algoritmo BPE

El algoritmo más extendido para construir un vocabulario de subpalabras es **Byte Pair Encoding (BPE)**: se parte del conjunto de entrenamiento troceado en caracteres individuales y, en cada iteración, se busca el par de tokens adyacentes más frecuente y se añade como un nuevo token al vocabulario. El proceso se repite hasta alcanzar el tamaño de vocabulario deseado. El resultado es un vocabulario en el que las palabras muy frecuentes acaban representadas por un único token, mientras que las palabras raras se descomponen en varias subpalabras más comunes.

La librería `tokenizers` de Hugging Face implementa este algoritmo (y sus variantes) de forma eficiente, permitiendo montar un tokenizador combinando tres piezas: el **modelo** de tokenización (el algoritmo en sí, p. ej. BPE), un **pre-tokenizador** (que decide cómo trocear el texto antes de aplicar el algoritmo, p. ej. separando por espacios) y un **entrenador**, que recorre el corpus para construir el vocabulario con el tamaño y los tokens especiales indicados, como `<unk>` para las palabras fuera de vocabulario o `<pad>` para el relleno de secuencias.

Una vez entrenado, el tokenizador puede usarse para codificar texto nuevo en tokens e ids, y también decodificar en sentido inverso; el objeto resultante conserva además los *offsets* de cada token (su posición exacta en el texto original), útiles para tareas que necesitan mapear las predicciones de vuelta al texto original.

### Preparando lotes: padding, truncamiento y attention mask

A diferencia del char-RNN, donde todas las ventanas tenían la misma longitud fija por construcción, las reseñas de IMDb tienen longitudes muy distintas entre sí. Para poder agruparlas en un tensor de lote es necesario que todas las secuencias tengan la misma longitud, lo cual se resuelve con dos mecanismos complementarios:

- **Padding**: rellenar las secuencias más cortas con un token especial (`<pad>`) hasta igualar la longitud de la secuencia más larga del lote (o de un máximo fijado).
- **Truncamiento**: cortar las secuencias que superen una longitud máxima.

El problema de rellenar con padding es que esos tokens no aportan información real, y el modelo debe saber ignorarlos. Para ello, junto a cada secuencia codificada el tokenizador genera una **attention mask**: un vector de unos y ceros (uno para los tokens reales, cero para el padding) que permite anular la contribución del padding, por ejemplo multiplicando las representaciones internas del modelo por esta máscara. A partir de esa misma máscara también es trivial obtener la longitud real de cada secuencia, simplemente sumando sus unos.

### Variantes de BPE: BBPE, WordPiece y Unigram LM

El BPE básico tiene un problema con los espacios: si el pre-tokenizador los elimina antes de trocear en caracteres, el tokenizador pierde la pista de dónde iban y puede acabar insertándolos en medio de una palabra al reconstruir el texto. La solución es **BBPE** (*byte-level BPE*), que sustituye los espacios por un carácter especial antes de tokenizar y, sobre todo, opera sobre los **bytes** del texto en lugar de sobre caracteres Unicode. Como cualquier texto —incluidos los emoticonos o símbolos poco comunes— puede descomponerse en un número finito de bytes (256 valores posibles), un vocabulario que contenga todos los bytes nunca necesita recurrir a un token desconocido: cualquier carácter, por raro que sea, siempre puede representarse a nivel de byte.

Existen otras variantes del mismo principio que cambian el criterio para decidir qué par de tokens fusionar, o qué tokens conservar:

| Característica | BBPE | WordPiece | Unigram LM |
|---|---|---|---|
| **Cómo funciona** | Fusiona los pares más frecuentes | Fusiona los pares que maximizan la verosimilitud de los datos | Elimina los tokens menos probables |
| **Ventajas** | Rápido, simple, ideal para multilingüe | Buen equilibrio entre eficiencia y calidad de los tokens | El más significativo, secuencias más cortas |
| **Desventajas** | Puede producir divisiones extrañas | Menos robusto que BBPE para multilingüe | Más lento de entrenar y tokenizar |
| **Usado por** | GPT, Llama, RoBERTa, BLOOM | BERT, DistilBERT, ELECTRA | T5, ALBERT, mBART |

**WordPiece**, introducida por Google en 2016, sigue el mismo esquema iterativo que BPE pero cambia el criterio de fusión: en vez de añadir el par más frecuente, añade el par con mayor puntuación según

$$
score(AB) = \frac{frequency(AB)}{frequency(A) \cdot frequency(B)}
$$

Esta puntuación favorece a los pares que aparecen juntos casi siempre, pero penaliza a los que, aun coincidiendo a menudo, también aparecen por separado en el resto del texto con frecuencia — es decir, prioriza fusiones realmente informativas frente a coincidencias casuales.

**Unigram LM**, por su parte, invierte el proceso: en vez de partir de caracteres sueltos e ir fusionando, arranca con un vocabulario muy grande que incluye prácticamente cualquier palabra, subpalabra y carácter frecuente del corpus, y va eliminando progresivamente los tokens menos útiles hasta alcanzar el tamaño deseado. Parte de la suposición de que el texto se ha generado muestreando tokens del vocabulario de forma independiente unos de otros, lo que le permite estimar qué tokens aportan menos a la verosimilitud del conjunto de entrenamiento y descartarlos.

Estas tres familias de tokenizadores no son alternativas exóticas: son, en la práctica, la capa de preprocesamiento que utilizan la mayoría de los grandes modelos de lenguaje actuales, y la elección de una u otra condiciona directamente el tamaño del vocabulario, la robustez ante distintos idiomas y la calidad de las representaciones que el modelo puede aprender a partir de esos tokens.

### Construyendo y entrenando un modelo de analisis de sentimientos

#### De generar texto a clasificarlo: modelos secuencia-a-vector

El char-RNN de la primera sección era un modelo **secuencia-a-secuencia**: producía una salida (un logit por carácter del vocabulario) en cada paso temporal, porque el objetivo era predecir el siguiente token en cada posición de la ventana. La clasificación de sentimiento es un problema distinto: dada una reseña completa, solo interesa una única predicción al final (positiva o negativa), no una por cada token leído. Esto define un modelo **secuencia-a-vector**: la GRU sigue procesando la reseña token a token igual que antes, pero de todas las salidas intermedias que produce solo se conserva el estado oculto final de la última capa (el resumen de haber leído toda la secuencia), que es el que se pasa a la capa de clasificación. El resto de salidas intermedias se descartan porque no aportan nada a una predicción que solo tiene sentido al terminar de leer la reseña entera.

#### Ignorando el padding desde la propia capa de embedding

La sección anterior ya explicaba cómo el padding y la *attention mask* resuelven el problema de agrupar reseñas de longitud distinta en un mismo lote. La capa de `Embedding` ofrece un mecanismo complementario para lidiar con ese padding: el argumento `padding_idx`. Al indicarle qué id corresponde al token de relleno, la capa fuerza a que ese vector de embedding concreto sea siempre cero y, además, deja de recibir gradiente durante el entrenamiento, de modo que nunca se aleja de cero por mucho que se entrene el modelo. La idea es que, si la entrada de la GRU en las posiciones de padding es un vector nulo, su influencia sobre el estado oculto debería ser mínima.

#### Clasificación binaria: una sola neurona de salida y `BCEWithLogitsLoss`

A diferencia del char-RNN, donde la capa lineal final tenía tantas salidas como caracteres en el vocabulario (para poder aplicar `softmax` y elegir entre muchas clases), aquí el problema es una clasificación **binaria**: solo hay dos clases posibles (reseña positiva o negativa), así que basta con una única neurona de salida. Ese único logit se interpreta como la evidencia a favor de la clase positiva: cuanto más alto (más positivo), más segura está la red de que la reseña es positiva; cuanto más bajo (más negativo), más segura de que es negativa. La función de pérdida adecuada para este planteamiento es `BCEWithLogitsLoss` (*binary cross-entropy with logits*), que aplica internamente una sigmoide al logit para convertirlo en una probabilidad entre 0 y 1 y calcula la entropía cruzada respecto a la etiqueta real, todo en una sola operación numéricamente más estable que encadenar una sigmoide manual con la entropía cruzada por separado.

#### El límite del `padding_idx`: por qué no basta con anular el vector de entrada

El `padding_idx` resuelve el problema solo a medias. Que el embedding de un token de padding sea el vector cero no significa que la GRU lo ignore: la capa recurrente sigue dando un paso por cada posición de la secuencia, padding incluido, actualizando su estado oculto en cada uno de ellos aunque la entrada de ese paso sea nula. Si una reseña es mucho más corta que el máximo del lote, acaba con muchos pasos "vacíos" al final, y el estado oculto final —justo el que se usa para clasificar— puede verse ligeramente distorsionado por esos pasos de más, en vez de ser exactamente el estado que había justo al terminar de leer el último token real.

#### Secuencias empaquetadas: eliminar el padding en vez de camuflarlo

La solución más rigurosa es no dejar que la RNN vea el padding en absoluto, en lugar de intentar minimizar su impacto. Para ello PyTorch ofrece las **secuencias empaquetadas** (`PackedSequence`), una estructura de datos que representa de forma compacta un lote de secuencias de longitudes distintas: en vez de un tensor rectangular con relleno, guarda únicamente los tokens reales de cada secuencia junto con la información necesaria para saber a qué secuencia y qué posición pertenece cada uno. La función `nn.utils.rnn.pack_padded_sequence()` construye esta estructura a partir de un tensor con padding y la lista de longitudes reales de cada secuencia del lote (la misma longitud que ya se obtenía sumando la *attention mask*); `enforce_sorted=False` evita tener que ordenar de antemano el lote de mayor a menor longitud, que es el orden que la función espera por defecto.

Las capas recurrentes de PyTorch (`GRU`, `LSTM`, `RNN`) aceptan un `PackedSequence` exactamente igual que un tensor normal, pero al procesarlo se detienen en cada secuencia justo en su último token real, sin dar pasos adicionales sobre el padding. Esto tiene dos efectos: es más eficiente, porque no se malgasta cómputo en pasos que no aportan información, y sobre todo es más correcto, porque el estado oculto final que se usa para clasificar queda garantizado como el estado justo después del último token real de cada reseña, en vez de una aproximación que depende de cuánto padding haya arrastrado el modelo detrás.

#### Comparando ambos enfoques

Entrenar ambas variantes del modelo —una con padding simple más `padding_idx`, otra con secuencias empaquetadas— partiendo de la misma semilla, los mismos hiperparámetros, el mismo optimizador (`Adam`) y la misma pérdida permite aislar el efecto real del padding sobre el desempeño final, en vez de una diferencia debida al azar de la inicialización o a otros hiperparámetros distintos. La comparación se hace con `BinaryAccuracy`, la métrica de precisión adecuada para un problema de clasificación binaria, evaluada sobre el mismo conjunto de validación para ambos modelos.

### RNNs bidireccionales

En cada uno de los pasos de tiempo, una capa recurrente solo mira a las entradas pasadas y presentes, antes de generar la salida; no es capaz de ver el futuro. Esta RNN tiene sentido cuando estamos haciendo prediccion en unas series temporales o en el decoder de un modelo seq2seq, pero para tareas como la clasificacion de texto o en el encoder de un modelo seq2seq normalmente es preferible mirar al texto que viene despues antes de codificar una palabra.

Una posible solucion es utilizar dos capas recurrentes sobre las mismas entradas, una leyendo las palabras de izquierda a derecha y otra al reves y combinar sus salidas en cada paso de tiempo, normalmente concatenandolas. Esto es precisamente lo que hacen las RNNs bidireccionales.

### Reutilizando embeddings y modelos preentrenados

Nuestros modelos han sido capaces de entrenar embedings utiles para miles de tokens, basados en el texto de las reseñas, aun que impresionante, si tuviese mucho mas texto podria reflejar de una mejor manera el texto incrustado. Por eso es bastante mas util utilizar embeddings preentrenados, aun que estos no hayan utilizado texto de la tarea que se intenta resolver. 

Los embeddings preentrenados llevan siendo populares mucho tiempo, aun que este acercamiento a la solucion tiene sus limites, en particular la representación de una sola palabra en diferentes contextos. La palabra "right" se codifica de manera exactamente igual en "left and right" y en "right and wrong", aun que tengan significados diferentes. Por esto, en un paper de 2018 [ https://homl.info/elmo ] se intenta dar una solución; Embeddings from Language Models (ELMo), los cuales son incrustaciones de palabras contextualizadas, aprendidas de los stados internos de RNNs profundas bidireccionales. En este caso, en vez de utilizar incrustaciones preentrenadas en el modelo, se reusan varias capas de un modelo del lenguaje preentrenado.

Mas o menos a la vez se presenta ULMFiT [ https://homl.info/ulmfit ] demostrando la efectividad del preentrenamiento sin supervision para tareas de PNL. Se entreno un modelo del lenguaje LSTM sobre un texto enorme utilizando aprendizaje auto supervisado y luego se hizo finetunning en varias tareas. Los autores demostraron que el ajustar un modelo preentrenado sobre solo 100 ejemplos etiquetados podia alcanzar el desempeño de uno entrenado sobre 10000 ejemplos.

### Trainer API

Hasta ahora, cada variante del modelo de sentimiento se ha entrenado con un bucle manual (`entrenar_nn`): tokenizar en el `collate_fn`, mover tensores al dispositivo, calcular la pérdida, hacer `backward()` y `step()`, evaluar en validación y decidir cuándo guardar el mejor modelo. La librería `transformers` ofrece una alternativa de más alto nivel para este mismo trabajo: la **`Trainer` API**, que encapsula ese bucle completo (incluyendo checkpoints, *early stopping*, entrenamiento distribuido en varias GPUs, registro de métricas y manejo de padding) detrás de una interfaz declarativa, a costa de perder parte del control fino que sí se tiene escribiendo el bucle a mano.

A diferencia del `collate_fn` usado hasta ahora, que tokenizaba las reseñas sobre la marcha en cada lote, la `Trainer` API espera trabajar con un `Dataset` ya tokenizado de antemano. Esto se consigue con el método `map()` de la librería `datasets`, que aplica una función de preprocesado a todo el conjunto; pasarle `batched=True` hace que esa función reciba lotes enteros de reseñas en vez de una a una, lo cual es mucho más rápido porque el tokenizador puede procesarlas en paralelo internamente.

El modelo también cambia de forma: en vez de construir a mano una GRU sobre los embeddings o los estados de BERT, se usa `BertForSequenceClassification.from_pretrained(...)`, una clase de `transformers` que ya incluye una cabeza de clasificación (una capa lineal) encima del BERT preentrenado, lista para afinarse (*fine-tuning*) sobre la tarea concreta indicando `num_labels`. Cargarlo con `dtype=torch.float16` almacena sus pesos en **precisión de 16 bits** en vez de los 32 bits habituales, reduciendo a la mitad la memoria que ocupa el modelo y acelerando el cómputo en GPUs que soportan esta precisión, a cambio de una pérdida de precisión numérica normalmente asumible.

El resto de piezas de la API tienen un equivalente directo en lo ya visto en la práctica:

- **`TrainingArguments`** centraliza los hiperparámetros de entrenamiento (número de épocas, tamaño de lote, cada cuánto evaluar/guardar/registrar). `load_best_model_at_end=True` junto con `metric_for_best_model="accuracy"` es el equivalente automático de la lógica de "guardar el mejor `state_dict` según la métrica de validación" que `entrenar_nn` implementaba a mano.
- **`compute_accuracy`**, la función que se le pasa a `Trainer` para evaluar, cumple el mismo papel que la métrica de `torchmetrics` (`BinaryAccuracy`) usada en el resto de la práctica.
- **`DataCollatorWithPadding`** sustituye al `collate_fn` manual: se encarga de aplicar padding dinámico a cada lote (rellenando solo hasta la reseña más larga de ese lote concreto, no de todo el dataset), la misma idea que resolvían a mano el tokenizador y `padding_idx`/las secuencias empaquetadas en las clases `SentimentAnalysisModel*`.

Con todas estas piezas montadas en un objeto `Trainer`, todo el bucle de entrenamiento (y la evaluación periódica en validación) queda reducido a una única llamada: `trainer.train()`.

### HF_pipelines

Si `Trainer` automatiza el *entrenamiento*, la función **`pipeline()`** automatiza el extremo opuesto: usar un modelo ya entrenado y afinado por otros, sin entrenar ni afinar nada en absoluto. Es el punto final de la progresión de esta práctica, que ha ido subiendo el nivel de reutilización paso a paso — modelo entrenado desde cero, luego solo los embeddings de BERT reutilizados, después una GRU afinada sobre BERT congelado, luego BERT completo afinado con `Trainer` — hasta llegar aquí, donde ni siquiera hace falta un paso de entrenamiento propio.

`pipeline("sentiment-analysis", model=...)` empaqueta en un único objeto el modelo, su tokenizador y todo el pre/postprocesado necesario para una tarea concreta; si no se especifica el argumento `model`, usaría el modelo por defecto asociado a esa tarea. Aquí se elige explícitamente `distilbert-base-uncased-finetuned-sst-2-english`: una versión reducida de BERT (**DistilBERT**) preentrenada sobre la Wikipedia en inglés y un corpus de libros, y afinada sobre SST-2 (Stanford Sentiment Treebank), una tarea de clasificación de sentimiento distinta de IMDb. Pese a no haber visto una sola reseña de IMDb durante su afinado, este modelo alcanza un 88.2% de precisión sobre el conjunto de validación de IMDb, una muestra de hasta qué punto puede generalizar un modelo de lenguaje bien preentrenado y afinado sobre una tarea relacionada.

El pipeline gestiona automáticamente varios detalles que en el resto de la práctica había que resolver a mano: usa la GPU si hay una disponible (con varias GPUs, se elige cuál usar mediante el argumento `device`), y los modelos de `transformers` se cargan siempre en modo evaluación por defecto, sin necesidad de llamar a `.eval()` como se hacía con los modelos propios. El resultado de aplicar el pipeline a un lote de reseñas es una lista con una etiqueta (`"POSITIVE"` o `"NEGATIVE"`) y un `score` por cada una: la probabilidad que el modelo asigna a *esa* etiqueta concreta, no a la clase positiva. Al tratarse de un problema binario, ese `score` nunca puede ser inferior a 0.5, porque si lo fuera el modelo habría elegido directamente la otra etiqueta.