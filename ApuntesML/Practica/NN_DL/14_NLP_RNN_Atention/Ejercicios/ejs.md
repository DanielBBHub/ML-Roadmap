# Ejercicios

### Ejercicio 1

¿Cuáles son las ventajas y los inconvenientes de usar una RNN con estado (*stateful*) frente a una RNN sin estado (*stateless*)?

Con estado
- 
Ventajas:
- Mayor contexto para la generación de la salida del modelo
- Poder procesar secuencias más largas

Desventajas:
- Mayor coste computacional
- Suposición de que los lotes son independientes e identicamente distribuidos
- Preparación más compleja del dataset (lotes consecutivos, sin barajar y reinicio de estado)

Sin estado
- 
Ventajas:
- Entrenamientos más rápidos
- Simplicidad de la arquitectura del modelo

Desventajas:
- Capacidad muy limitada para procesar secuencias, unicamente la longitud de ventana definida

### Ejercicio 2

¿Por qué se usan RNNs encoder-decoder en lugar de RNNs secuencia-a-secuencia simples para la traducción automática?

Las RNNs encoder-decoder tienen las características de poder encapsular los tokens del idioma original en un vector que los represente para luego poder generar palabra a palabra la traducción al idioma objetivo. Es decir, hay un paso de representación general del significado de la frase original y un paso de representación especifica del significado de la frase en el idioma objetivo que las RNNs seq2seq normales no pueden alcanzar debido a su arquitectura base; estas traducirian literalmente palabra por palabra sin tener en cuenta el resto de la oración, en cuanto leen la primera palabra.

Estas características se acentúan con el uso de "teacher forcing", "beam search" o mecanismos de atención, que mitigan los problemas de entrenamiento, longitudes de secuencias y "memória" del contexto de la secuencia.

### Ejercicio 3

¿Cómo se pueden manejar secuencias de entrada de longitud variable? ¿Y secuencias de salida de longitud variable?
Las secuencias de entrada de longitud variable se pueden manejar de varias maneras:

- Padding con tokens predefinidos (como 0s) + mascaras de atención

- Truncamiento de secuencias en una longitud fija

- Forzar secuencias de tamaños similares 

- Packed Sequences (secuencias empaquetadas)

Por otro lado, las secuencias de salida con longitudes variables se pueden manejar de las siguientes maneras:

- Utilizar tokens de principio y final de secuencia (<s> y </s>)

- Forzar una longitud de salida

### Ejercicio 4

¿Qué es la búsqueda en haz (*beam search*) y por qué la usarías? ¿Qué herramienta puedes usar para implementarla?

La busqueda de haz, o "beam search", es un método para mejorar el desempeño del modelo encoder-decoder, el cual se basa en generar varias posibles secuencias de salida en base a probabilidades estimadas sobre una palabra generada. Este método genera k posibles secuencias candidatas que van acumulando probabilidades estimadas y van filtrando las k mejores, con el objetivo de que un error al principio del proceso no resulte en una generación erróne a al final. Esta se podría utilizar en traducciones autómaticas. El parametro k indica la anchura del haz; cuanto más grande es, más secuencias candidatas valora, mejor traducciones genera pero requiere más cómputo y memória.

Se puede utilizar implementando una celula de memoria personalizada para que realice esta función.

### Ejercicio 5

¿Qué es un mecanismo de atención? ¿En qué ayuda?

Un mecanismo de atención es un componente de la arquitectura encoder-decoder que, en cada paso de generación del decoder, calcula una combinación ponderada de todos los estados ocultos del encoder (no solo del último), en lugar de depender de un único vector de contexto fijo. Estos pesos, uno por cada token de la entrada, se obtienen aplicando una función de alineamiento (por ejemplo un producto escalar o una pequeña red densa) entre el estado oculto actual del decoder y cada estado oculto del encoder, normalizándolos después con un softmax. El resultado es un vector de contexto dinámico, distinto en cada paso, que indica a qué palabras de la entrada debe "prestar atención" el decoder para generar el siguiente token.

Este mecanismo ayuda principalmente porque elimina el cuello de botella de tener que comprimir toda la secuencia de entrada en un único vector de tamaño fijo (el problema clásico de las RNN encoder-decoder simples), lo que mejora notablemente el rendimiento en secuencias largas. Además, al dar al decoder acceso directo a los estados ocultos del encoder en cada paso, facilita el flujo del gradiente durante el entrenamiento y suele acelerar y estabilizar la convergencia. Como efecto añadido, los pesos de atención son interpretables: permiten visualizar qué partes de la entrada influyeron más en cada palabra generada.

### Ejercicio 6

¿Cuándo necesitarías usar *sampled softmax*?

El sampled softmax es una técnica que se usa durante el **entrenamiento** (no en la inferencia) cuando el vocabulario de salida es muy grande (decenas o cientos de miles de clases), como suele ocurrir en modelos de lenguaje o en el decoder de un traductor automático. Calcular el softmax completo en cada paso implica evaluar los logits de todas las clases del vocabulario, lo cual resulta muy costoso en cómputo y en memoria.

El sampled softmax aproxima esa pérdida softmax completa muestreando, en cada paso de entrenamiento, solo un pequeño subconjunto de clases negativas (además de la clase correcta), en lugar de usar todo el vocabulario. Así se reduce drásticamente el coste de calcular la pérdida y sus gradientes, acelerando mucho el entrenamiento con una pérdida de precisión mínima.

Por tanto, se necesitaría usar cuando se entrena un modelo con un vocabulario de salida muy grande y el cálculo del softmax completo se convierte en el cuello de botella del entrenamiento. En inferencia no se usa, ya que ahí sí interesa obtener las probabilidades (o logits) de todas las clases para elegir el token más probable o aplicar *beam search*.

### Ejercicio 7

Hochreiter y Schmidhuber usaron las gramáticas de Reber embebidas (*embedded Reber grammars*) en su artículo sobre las LSTM. Son gramáticas artificiales que producen cadenas como "BPBTSXXVPSEPE". Consulta la buena introducción de Jenny Orr a este tema, elige una gramática de Reber embebida concreta (como la que aparece representada en la página de Orr) y entrena una RNN para identificar si una cadena respeta esa gramática o no. Primero tendrás que escribir una función capaz de generar un lote de entrenamiento que contenga aproximadamente un 50 % de cadenas que respeten la gramática y un 50 % que no.

### Ejercicio 8

Entrena un modelo encoder-decoder capaz de convertir una fecha escrita en un formato a otro formato (por ejemplo, de "April 22, 2019" a "2019-04-22").
