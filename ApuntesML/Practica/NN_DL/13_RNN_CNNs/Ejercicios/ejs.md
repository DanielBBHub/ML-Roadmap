# Ejercicios

## 1. ¿Se te ocurren algunas aplicaciones para una RNN secuencia-a-secuencia? ¿Y para una RNN secuencia-a-vector, y una vector-a-secuencia?
Aplicaciones de Seq2Seq:
- Predicción de pasajeros en un servicio de trenes 
- Generación de audio

Aplicaciones de Seq2Vec:
- Codificado de un archivo de voz
- Análisis del sentimiento de una reseña

Aplicaciones Vec2Seq:
- Decodificado a un archivo de voz
- Estilizado de caracteres (esto solo sería correcto si la entrada es un identificador de carácter y no un trazo del usuario)


## 2. ¿Cuántas dimensiones deben tener las entradas de una capa RNN? ¿Qué representa cada dimensión? ¿Y sus salidas?
La entrada de una capa de RNN es un tensor con un total de tres dimensiones: [(tamaño_lote), (paso_temporal), (muestra)]. La primera dimensión define el tamaño del lote que se esta utilizando, la segunda indica la posición en la secuencia / el paso temporal de la muestra y la tercera es la muestra.

La salida puede ser un vector o una secuencia, con lo que puede tener dos posibilidades:
- Vector: [(tamaño_lote), (lista_muestras)]
- Secuencia: [(tamaño_lote), (paso_temporal), (muestra)]

## 3. ¿Cómo se puede construir una RNN secuencia-a-secuencia profunda en PyTorch?
Una RNN secuencia-a-secuencia profunda se puede construir simplemente con el modulo nn.RNN modificando el valor del argumento "num_layers" > 1, esto creará tantas capas RNN como se haya definido. Luego se define una capa densa para producir el resultado de la inferencia del modelo. 

## 4. Supón que tienes una serie temporal univariante diaria y quieres pronosticar los próximos siete días usando una RNN. ¿Qué arquitectura deberías usar?
La arquitectura que debería utilizarse en este caso es una arquitectura Seq2Seq, ya que tanto la entrada como la salida serían secuencias, en este caso series temporales univariantes. La entrada con la que se entrena tiene una unica variable ordenada por el paso del tiempo y el objetivo de la red sería producir un resultado que predijese que valores tendrían los siguientes siete pasos temporales

## 5. ¿Cuáles son las principales dificultades al entrenar RNNs? ¿Cómo se pueden manejar?
Las dificultades al entrenar RNNs son dos:
- Problemas de gradientes inestables, ya sea que explotan o desaparecen
- La memória a corto plazo

#### Problemas de gradientes inestables
Para apaliar este problema se pueden tomar varias medidas:
- Buena inicialización de los pesos de la red
- Mejores optimizadores 
- Utilizar funciones de activación saturables (como tanh) para evitar la explosión de los gradientes
- Utilizar técnicas de normalización como: Batch Norm, Layer Norm o Dropout

#### Memória a corto plazo
Para este aspecto se desarrollaron dos tipos de células de memória:

- LSTM (long short-term memory)
- GRU

Con lo que habría que utilizar estas células en la implementación de la arquitectura de la red. Aun que si la entrada son secuencias muy largas, no sería suficiente y seguirían perdiendo el contexto del principio de la secuencia, con lo que podríamos añadir otro tipo de capa a la arquitectura. 

Las capas que podrían añadirse son capas convolucionales 1D, para procesar las secuencias. Estas, debido a que procesan la información mediante kernels/ventanas, lograrían comprimir la entrada para hacer una secuencia más corta, la cual recibiría una capa recurrente.

## 6. ¿Puedes esbozar la arquitectura de la celda LSTM?
La arquitectura de la celda LSTM parte conceptualmente de la RNN básica, en el sentido de que tiene estados ocultos y una salida. En base a esto, en LSTM se implementó la separación del estado oculto de la RNN básica en dos, un $h_{(t)}$ y un $c_{(t)}$ que se encargarían de la memória a corto y largo plazo, respectivamente. El estado $c_{(t)}$ dentro de la célula se verá multiplicado elemento a elemento para eliminar partes del estado oculto, en caso de que $f_{(t)}$ saque como output un 0. Más adelante se le sumarán partes de la entrada que genera la capa $g_{(t)}$ que elige la capa $i_{(t)}$ al multiplicar elemento por elemento el resultado de la anterior capa para, seguidamente, hacer una copia del estado $c_{(t)}$, multiplicarlo elemento por elemento con el resultado de la capa $o_{(t)}$ que elegirá que información ha de permanecer en el estado de corto plazo $h_{(t)}$. Este proceso se repite cada paso temporal del entrenamiento y es capaz de conseguir aprender que información es neecsaria, cual ha de borrar y cual ha de sacar por la celula para reconocer patrones temporales en las secuencias de entrada

## 7. ¿Por qué querrías usar capas convolucionales 1D en una RNN?
La utilidad de utilizar capas convolucionales 1D en una RNN es debido a que son capaces de comprimir las secuencias sin perder mucha información, aplicando filtros con anchura variable sobre las secuencias, lo que, a parte de reducir el tamaño de las secuencias en si, permite a las RNN abarcar un período de tiempo más grande y captar patrones en plazos de tiempo mayores

## 8. ¿Qué arquitectura de red neuronal podrías usar para clasificar vídeos?
La arquitectura de la red neuronal que se podría utilizar sería una combinación de capas convolucionales 2D para reducir la dimensión de las imágenes y luego utilizar una arquitectura seq2vec para generar una salida que clasificase el video procesado

## 9. Intenta modificar el modelo Seq2SeqModel para pronosticar tanto el tráfico de tren como el de autobús para los próximos 14 días. El modelo ahora necesitará predecir 28 valores en lugar de 14.

## 10. Descarga el conjunto de datos de corales de Bach y descomprímelo. Está compuesto por 382 corales compuestos por Johann Sebastian Bach. Cada coral tiene entre 100 y 640 pasos de tiempo de longitud, y cada paso de tiempo contiene 4 enteros, donde cada entero corresponde al índice de una nota en un piano (excepto el valor 0, que significa que no se toca ninguna nota). Entrena un modelo —recurrente, convolucional, o ambos— que pueda predecir el siguiente paso de tiempo (cuatro notas), dada una secuencia de pasos de tiempo de un coral. Luego usa este modelo para generar música al estilo de Bach, una nota a la vez: puedes hacerlo dando al modelo el comienzo de un coral y pidiéndole que prediga el siguiente paso de tiempo, luego añadiendo estos pasos de tiempo a la secuencia de entrada y pidiendo al modelo la siguiente nota, y así sucesivamente. Asegúrate también de revisar el modelo Coconet de Google, que se usó para un bonito doodle de Google sobre Bach.

## 11. Entrena un modelo de clasificación para el conjunto de datos QuickDraw, que contiene millones de bocetos de varios objetos. Empieza descargando los datos simplificados de unas pocas clases (por ejemplo, ant.ndjson, axe.ndjson y bat.ndjson). Cada archivo NDJSON contiene un objeto JSON por línea, que puedes analizar usando la función json.loads() de Python. Esto te dará una lista de bocetos, donde cada boceto se representa como un diccionario de Python. En cada diccionario, la entrada "drawing" contiene una lista de trazos del lápiz. Puedes convertir esta lista en un tensor 3D de flotantes donde las dimensiones son [trazos, coordenadas x, coordenadas y]. Dado que una RNN toma una única secuencia como entrada, necesitarás concatenar todos los trazos de cada boceto en una única secuencia. Es mejor añadir una característica extra para que la RNN sepa cuánto ha avanzado en cada trazo actualmente (por ejemplo, de 0 a 1). En otras palabras, el modelo recibirá una secuencia donde cada paso de tiempo tiene tres características: las coordenadas x e y del lápiz, y la proporción de progreso a lo largo del trazo actual.

## 12. Crea un conjunto de datos que contenga grabaciones de audio cortas de ti diciendo "sí" o "no", y entrena una RNN de clasificación binaria con él. Por ejemplo, podrías:
### a. Usar un software de grabación de audio como Audacity para grabarte diciendo "sí" tantas veces como tu paciencia te lo permita, con pausas cortas entre cada palabra. Crea una grabación similar para la palabra "no". Intenta cubrir las diversas formas en que podrías pronunciar realmente estas palabras en la vida real.

### b. Cargar cada archivo WAV usando la función torchaudio.load() de la biblioteca TorchAudio. Esto devolverá un tensor que contiene el audio, así como un entero que indica el número de muestras por segundo. El tensor de audio tiene una forma de [canales, muestras]: un canal para mono, dos para estéreo. Convierte el estéreo a mono promediando sobre la dimensión de canal.

### c. Trocear cada grabación en palabras individuales dividiendo por los silencios. Puedes hacer esto usando la transformación torchaudio.transforms.Vad (Detección de Actividad de Voz).

### d. Dado que las secuencias son tan largas, es difícil entrenar directamente una RNN con ellas, así que ayuda convertir primero el audio en un espectrograma. Para esto, puedes usar la transformación torchaudio.transforms.MelSpectrogram, que es muy adecuada para la voz. La salida es una secuencia dramáticamente más corta, con muchos más canales.

### e. ¡Ahora intenta construir y entrenar una RNN de clasificación binaria con tu conjunto de datos de sí/no! Considera compartir tu conjunto de datos y modelo con el mundo (por ejemplo, a través de Hugging Face Hub).