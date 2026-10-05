# Transformers for Natural Language Processing and Chatbots

En un paper de 2017 [ https://arxiv.org/abs/1706.03762 ] "Attention is all you need", un equipo de investigadores de Google propuso una nueva arquitectura de redes neuronales llamada "Transformer", la cual mejoraba sustancialmente la traducción automática SotA (state of the art). Esencialmente, era una arquitectura encoder decoder.

1. El texto original entraba al decoder, el cual generaba un embedding contextualizado (uno por token)
2. La salida del encoder se utiliza como entrada del decoder, así como el texto traducido hasta ahora (con un token de SoS)
3. El decoder predice el siguiente token para cada token de entrada
4. El último token de salida del decoder se agrega a la traducción 
5. Los pasos 2,3 y 4 se repiten una y otra vez hasta producir una traducción completa, un token cada vez, hasta generar un token de final de secuencia. En fase de entrenamiento, como ya se tiene la traducción completa, se le da al decoder en el paso 2, con lo que los pasos 4 y 5 no son necesarios

La diferencia yace dentro de la caja negra que es la arquitectura, ya que el Transformer no contiene ninguna capa recurrente o convolucional, únicamente capas densas combinadas utilizando un nuevo mecanismo de atención: MHA (multi-head attention), además de algún que otro truco. Gracias a que contiene solamente capas densas, no sufre de los problemas de explosión/desaparición de gradientes, se puede entrenar en menos pasos y puede paralelizarse la ejecución en múltiples GPUs, además de escalar sorprendentemente bien. Además, gracias al nuevo mecanismo de atención, es capaz de captar patrones en rangos más grandes.

> Es importante destacar que, aún que se planteó inicialmente para la traducción automática, ha resultado ser mucho más versatil, dando a la explosión del NLP, con lanzamientos como los chatbots de OpenAI, Anthropic, ... Pero no acaba ahí, ya que también son útiles para visión artificial, procesado de audio, robótica, ...

En este capítulo se seguirán los siguientes putnos:

- Diseccionar la arquitectura original de los Transformers para entender como funcionan
- Implementar y entrenar un transformer para la traducción automática de inglés a castellano
- Inspeccionar modelos encoder como BERT, aprender como se han preentrenado y como utilizarlos para tareas como clasificación de texto, búsqueda semántica y agrupado de texto, con y sin fine-tuning
- Inspeccionar modelos decoder como GPT, aprender como se han preentrenado. Estos modelos son capaces de generar texto, además de otras tareas
- Implementar un modelo decoder para construir un chatbot
- Estudiar modelos encoder-decoder para tareas como traducción y resumido de texto

## Attention is all you need: la arquitectura original del transformer

Como ya hemos visto, el objetivo del encoder es transformar gradualmente la entrada hasta que cada representación de los tokens capture perfectamente el significado en el contexto de la oración, generando una secuencia de incrustaciones contextualizadas de tokens. 

El rol del decoder es recoger las salidas del encoder, acompañadas de la traducción hasta este punto y predecir el token siguiente. Para esto, las capas del decoder transforman gradualmente cada representación de los tokens de entrdad en una representación que puede usarse para predecir el siguiente token.

Los tensores que reciben los encoders y decoders son los siguientes:
- **[batch size, max English sequence length in the batch, embedding size]**: al iterar la entrada original se transforman los tokens en una representación contextualizada del significado
- **[batch size, max Spanish sequence length in the batch, embedding size]**: al iterar la entrada de la representación de tokens conjuntamente con la traducción hasta el momento, se logra la predicción del siguiente token

![Arquitectura transformers](img/Arquitectura%20transformers.png)

Vale la pena resaltar unas cuantas cosas de la arquitectura:

- Tanto el encoder como el decoder contienen bloques que están apilados N veces, en el paper en concreto $N=6$, también que las salidas de la pila entera de encoder se utilizan como entrada en los N bloques del decoder.

- La mayoría de las capas ya son conocidas: hay dos capas de embedding, varias skip connections seguidas de modulos de normalización, varios modulos compuestos por dos capas densas (la primera con función ReLU y la segunda sin) y finalmente la salida es una capa lineal. Como cada capa ya mencionada procesa cada token por separado, es donde entran los nuevos componentes de la arquitectura:
    - La capa MHA del encoder actualiza cada representación de los tokens centrandose en cada token en la misma oración, excluyendolo a él, es decir, utiliza "self-attention". De esta manera las palabras con representaciones vagas como "like" adquieren significado en cada oración por separado
    - La MMHA(masked MHA) hace lo mismo que la MHA pero cuando ya ha procesado un token, ya no hace caso a los tokens que van después de él; es una capa "causal" (cuando procesa el token "gusta" de la frase "me gusta el fútbol" solo hace caso a los tokens "<s>" "me" "gusta" e ignora "el" y "fútbol")
    - La capa MHA del decoder es donde este presta atención a las representaciones de los tokens generadas por el encoder. Esto se conoce como "cross-attention".
    - Los encodings posicionales son vectores densos que representan la posición de cada token en la oración. El enésimo encoding posicional es añadido al token incrustado del enésimo token de cada oración. Esto es necesarío por que todas las capas de la arquitectura del transformer tratan todas las posiciones por igual: cuando procesan un token no tienen idea de donde viene o su relación con el resto de tokens

### Encodings posicionales

Como se ha explicado antes, un encoding posicional es un vector denso que codifica la posición de un token en relación a una oración: el enésimo encoding posicional es añadido (se suma) al token incrustado en la enésima posición en cada frase.

$$
\begin{align}
\text{input\_embedding}(i) &= \text{token\_embedding}(i) + \text{positional\_encoding}(i) \\
X_i &= E_i + P_i
\end{align}
$$

donde:

- $E_i$ = embedding del token en la posición $i$
- $P_i$ = codificación posicional de la posición $i$
- $X_i$ = entrada final que recibe el modelo

### Atención multicabeza

La capa MHA está basada en el la atención de producto punto escalado, una variante de la atención de producto punto que escala hacia abajo la similitud de las puntuaciones por un factor constante 

$$
\large
\text{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}
$$

Donde:
- $Q$ es la matriz que representa la query (p.j. una frase en ingles). Su forma es $[L_q, d_q]$ donde $L_q$ es la longitud de la query y $d_q$ es la dimensionalidad de la query
- $K$ es la matriz representando una clave. Su forma $[L_k, d_k]$ donde $L_k$ es la longitud de la clave y $d_k$ es la dimensionalidad de la clave
- $V$ es una matriz representando un valor. Su forma $[L_v, d_v]$ donde $L_v$ es la longitud del valor y $d_v$ es la dimensionalidad del valor.
- La forma de $QK^T$ es $[L_q, L_k]$: contiene una puntuación de similitud para cada para query-clave. Para evitar que la matriz sea enorme, la secuencia de entrada no tiene que ser demasiado larga, ya que se toparía con el problema de la ventana de contexto cuadrática. La función softmax se aplica a cada fila, resultando en una forma $[L_q, d_v]$, donde hay una fila por cada token de la query y cada fila representa el resultado de la query: una suma ponderada de los valores de los tokens, favoreciendo los valores de los tokens cuyos tokens clave se alinean más con el token de la query
- El facto de escalado es $\frac{1}{\sqrt{d_k}}$ escala hacia abajo la similitud de las puntuaciones para evitar sturar la función softmax, lo cual llevaria a gradientes muy pequeños.
- Es posible enmascarar algunos pares clave-valor añadiendo valores negativos muy grandes a las puntuaciones de similitud correspondiende antes de calcular el softmax

Ahora miraremos a la capa de atención con multiples cabezas.

![capa de atención con multiples cabezas](img/capa%20de%20atención%20con%20multiples%20cabezas.png)

Como se puede ver son una pila de capas de atención de productos punto escalados llamadas cabezas de atención, cada una precedida por una transformación lineal de valores, llaves y querys. La salida de todas las cabezas de atención son concatenadas y pasan por una transformación lineal final.

Ahora viene el porque: en una oración como "I like football", "like" tiene que ser codificada por el encoder de manera que guarde todo el significado en el contexto de la frase, asi como su posición. El tema es que si solo utilizasemos una capa de atención, tendriamos que codificar todas las características en un solo vector, con lo que las capas separan estas representaciones de query,key,value en multiples cabezas, para que cada una pueda centrarse en características especificas del token. La "decisión" de que cabeza se ocupa de que la toma la primera capa lineal, mientras que la capa lineal final se encarga de reorganizar la representación de las características como decida.

## Transformers de solo codificacion para entendimiento del lenguaje natural

Cuando, en 2018 [ https://homl.info/bert ], Google lanzo BERT, probó al mundo que los transformers solo de codificación pueden realizar un amplio abanico de tareas, como clasificación de oraciones, tokens, resolver preguntas test, ... Además de confirmar la efectividad del preentrenamiento autosupervisado sobre un gran bloque de texto para transfer learning.

### Arquitectura de BERT

La arquitectura de BERT es casi identica al transformer encoder original, con tres grandes diferencias:
1. Tamaño. El modelo base tiene 12 bloques de encoders, 12 cabezas de atención y embeddings de 768 dimensiones. También utiliza embeddings posicionales entrenables y soporta frases de entrada de hasta 512 tokens.

2. Aplica LN justo antes de cada subcapa (sea atención o lineal), lo que se conoce como pre-LN, ya que se asegura que las entradas de cada subcapa esten normalizadas, así estabilizando el entrenamiento y reduciendo la sensibilidad a la inicialización de pesos.

3. Permite separar las oraciones de entrada en dos segmentos. Esto es util para tareas que requieran una pareja de oraciones de entrada, como inferencia sobre lenguaje natural o contestar preguntas de respuesta múltiple. Esto se puede conseguir utilizando un token de separación en cada oración y luego concatenarlas.

### Preentrenamiento de BERT

Se propusieron dos tareas de preentrenamiento autosupervisado

**Masked languaje model (MLM)**
-
Cada token en una oración tiene un 15% de probabilidades de ser reemplazado con un token de máscara y el modelo es entrenado para predecir cuales eran los tokens originales. Si tenemos la oración "Se lo pasó genial en el cumpleaños" la frase que le entraría al modelo sería "Se lo pasó [mascara] en [mascara] cumpleaños" y se calcularía la pérdida sobre estos tokens mascara. Para concretar, algunos de los tokens enmascarados no son realmente así; 10% son reemplazados por tokens aleatorios y otro 10% no se reemplazan por nada. Esto ayuda a que el modelo tenga un buen desempeño aun que no haya tokens y en cuanto a los tokens que no se reemplazan, estos hacen la predicción trivial, lo cual hace que el modelo "preste atención" al token de entrada que está en la posición del token predecido.

**Next sentence prediction (NSP)**
-
Se entrena al modelo para predecir si dos frases estan seguidas una de la otra. Esta es una tarea de clasificación binaria, en la cual se implementó un nuevo token de clase [CLS]: este token se inserta al principio (posición 0, segmento 0) y durante el entrenamiento del encoder este token se pasa por una cabeza de clasificación binaria.

Se entrenó BERT con estos dos métodos a la vez, con un gran volumen de textos, con el objetivo de que con NSP se obtuviese una buena representación de la frase de entrada con el embedding contextualizado de tokens, aunque luego se demostró que calcular las medias de los embeddings contextualizados generaba mejores resultados.

### Ajustando(fine-tuning) BERT

BERT se puede ajustar para realizar una retaila de tareas, cambiando muy poco para cada una de ellas.

Para tareas de clasificación, como analisis del sentimiento, todos los tokens de salida se ignoran excepto el primero, el cual corresponde al token de la clase, para luego reemplazar la cabeza de clasificacion NSP por una nueva. Luego puedes ajustar el modelo utilizando cross entropy, con un LR más bajo para las capas inferiores o congelandolas por completo las primeras iteraciones del entrenamiento para entrenar únicamente la cabeza clasificadora. Utilizando el mismo método se pueden abordar otras tareas de clasificación, como por ejemplo clasificar oraciones gramaticamente correctas con el dataset CoLA, por ejemplo.

Para clasificación de tokens, la cabeza de clasificación se aplica a cada token. Este modelo puede ser ajustado para "named entity recognition" (NER), donde el modelo etiqueta partes del textro que corresponden a nombres, fechas, lugares, organizaciones u otras entidades, lo cual puede aplicarse en terreno legal, financiero o medico. El mismo método puede utilizarse para otras tareas de clasificación de tokens, como etiquetar errores gramaticales, análisis del sentimiento a nivel de tokens, análisis sintáctico o encontrar preguntas, citas, saludos, ...

BERT también se puede utilizar para clasificar pares de secuencias, funcionando exactamente como clasificación de secuencias, pero en este caso con un par, en vez de una. Por ejemplo, esto puede utilizarse para "natural language inference" (NLI), donde el modelo debe determinar si la frase A conlleva la B, la contradice o ninguna de las dos. También puede utilizarse para detetcar si dos frases tienen el mismo significado, se estan parafraseando o si la respuesta a la pregunta A está presente en la frase B.

Para preguntas de respuesta múltiple se llama a BERT por cada respuesta posible, poniendo la pregunta en el segmento 0 y la posible respuesta en el 1. Para cada respuesta la el token de salida con la clase es pasado por una capa lineal con una sola unidad, produciendo una puntuación. Una vez se han procesado todas las respuestas se convierten en probabilidades utilizando la capa softmax. Se puede utilizar cross entropy para ajustar el modelo.

También es hábil en "extractive question answering"; se le pregunta al modelo (segmento 0) sobre un texto llamado "contexto"  (segmento 1) y BERT tiene que descubrir donde está la respuesta en el contexto. Para esto se puede añadir una capa lineal con dos unidades encima de BERT para generar uns puntuaciones por token: una puntuación de inicio y otra de final. Durante el ajuste se pueden tratar como logits para dos clasificadores binarios, el primero clasifica si es el inicio y el otro si es el final de la respuesta. Por supuesto, la mayoría no será ninguno y puede que el mismo sea el inicio y el final si la respuesta es un único token. En tiempo de inferencia se selecciona el par de indices $i$ y $j$ que maximizan la suma del token de inicio $i$ y de final $j$, sabiendo que $i \leq j$ y $j - i + 1 \leq \text{ len respuesta }$.

Los autores también demostraron que BERT podia ser ajustado para medir "semantic textual similarity" (STS). En el conjunto STS-B se le da al modelo dos oroaciones y saca una puntuació que indica como de similares son semánticamente. Como esto fuerza a BERT a correr en $O(N^2)$ es preferible utilizar SBERT (Sentente-BERT), el cual es una variante de BERT que se ajusto para producir mejores embeddings de oraciones. Se empieza generando el embedding para después medir la similitud con la medición de la similitud por coseno ($[-1,1]$ dependiendo de lo diferentes/iguales que sean). 

Por otro lado, el embedding de oraciones puede ser muy útil en otras aplicaciones:

*Agrupación de texto*  
    Pueden procesarse un gran número de domcumentos con SBERT para obtener los embeddings y aplicar un algoritmo de agrupación como k-means o HDBSCAN para agrupar los documentos basándote en la similitud semántica de estos  
*Búsqueda semántica*  
    Puedes ayudar al usuario a encontrar documentos basados en el significado de la query. Se codifican los documentos utilizando SBERT y se guardan los embeddings para depués comparar el codificado de la query con los de los documentos.  
*Reordenación de resultados de búsqueda*  
    Pueden reordenarse los resultados de una búsqueda basandote en la similitud semántica con la query  

### Otros modelos de solo codificación

#### RoBERTA - Facebook AI (125-355M parámetros)

Similar a BERT pero con mejor desempeño en general, en gran parte por que su preentrenamiento fue más largo y sobre un dataset mayor. Se utilizó MLM pero no NSP. Es importante destacar que se utilizó "dynamic masking", es decir, los tokens se enmascararon durante el entrenamiento, con lo que el mismo extracto de texto se enmascara de manera diferente en diferentes épocas. Esto consigue que el modelo tenga más diversidad en la información, reduzca el overfitting y tenga una mejor generalización

#### DistilBERT - Hugging Face (66M parámetros)

Un modelo más pequeño (40%) y más rápido (60%) que consigue llegar al 97% de desempeño de BERT en la mayoría de tareas, haciéndolo una buena opción para entornos con bajos recursos, aplicaciones que necesitan poca latencia o un ajustado rápido.

Como su nombre indica, el modelo fue entrenado con una técnica llamada "model distillation", propuesta en un paper de 2015 [ https://homl.info/distillation ]. Esta es básicamente entrenar un pequeño modelo "estudiante" con las probabilidades estimadas de un modelo "profesor" como etiquetas. Estos son "soft targets", más alla de los vectores one-hot usuales: hace que los entrenamientos sean mucho más rápido y la información más eficiente, ya que consigue que el modelo estudiante aprenda directamente de la distribución correcta, en vez de tener que aprenderla durante entrenamiento.

Es importante saber que las probabilidades estimadas para el alumno y el profesor se suavizan durante entrenamiento dividendo los logits finales por una temperatura mayor a 1 (normalmente 2). Esto dota al estudiante con señales más complejas que cubren todas las posibles opciones, en vez de centrarse en la respuesta correcta. El autor bautizó este concepto como "dark knowledge". Si la frase fuese "Hace sol y me encuentro [mask]" las probabilidades podrian ser un 72, 27 y 0.5 para "genial", "bien" y "mal", con lo que para el estudiante serían un 60, 36 y 5% respectivamente. Siempre es útil que el modelo sepa que "mal" es una opción plausible aun que sea rara.

La périda en el entrenamiento tiene dos componentes más; la pérdida estándar MLM y la cosine embedding loss, que minimiza la similitud de coseno entre los estados finales del alumno y el profesor, permitiendo que el estudiante "piense" como el profesor en vez de hacer las mismas predicciones, además de llegar más rapido a la convergencia y mejorar el desempeño.

#### ALBERT - Google Research (12-235M parámetros)

Todas las capas de encoder de este modelo comparten pesos, haciendo que este sea mucho más pequeño que BERT, pero no más rápido. Es un buen modelo para cuando la mémoria escasea, en particular cuando estás entrenándolo en una GPU con poca VRAM.

También se introdujo el "factorized embeddings" para reducir el tamaño de la capa de incrustación: en BERT-large el tamaño de vocabulario era aproximadamente de 30000 y el tamaño del embedding de 1024, lo que significa que la matriz de embedding tenia más de 30 millones de parámetros. ALBERT reemplaza esta matriz con el producto de dos más pequeñas. En la practica se puede implementar reduciendo el tamaño de embedding - ALBERT utiliza 128 - y añadiendo una capa lineal despues de la de embedding para proyectarlos a un espacio dimensional mayor, como 1024 dimensiones para ALBERT-large. 

Por otro lado, también se cambió NSP por "sentence order prediction" (SOP) que, dadas dos oraciones consecutivas, el objetivo es predecir cual va primero. Es una tarea más dificil que NSP pero conllevó una mejora de la incrustación de las secuencias

#### ELECTRA - Google Research (14-335M parámetros)

Este modelo introdujo una nueva técnica de preentrenamiento, "replaced token detection" (RTD): se entrenaron dos modelos juntos - un pequeño modelo generador y un modelo más grande que discriminaba. El generador solo se utiliza durante el entrenamiento, mientras que el discriminador es el que se acabará utilizando. El primero se entrena utilizanodo MLM con enmascarado dinámico; para cada token enmascarado un token de reemplazo se escoge de las mejores predicciones. El texto resultado se entrega directamente al discriminador, que tiene la tarea de predecir que token es original y cual no.

Esta técnica es más eficiente con las muestras que MLM ya que el discriminador aprende sobre más tokens por ejemplo, convergiendo más rápido y, generalmente, con un desempeño igual a modelos como BERT-large. Dicho esto, puede que los beneficios no compensen la complejidad adicional en el preentrenamiento.

#### DeBERTa - Microsoft (139M-1.5B parámetros)

Este es un modelo relativamente grande, que mejoró los modelos SotA en muchas tareas de entendimiento del lenguaje natural. Elimina la capa de embedding posicional y utiliza "relative positional embedding" cuando calcula la puntuación de antención dentro de cada capa de atención multicabeza: cuando se decide cuanto debería atender la i-query al j-token, el modelo tiene acceso a un embedding aprendido para la posición relativa i-j. DeBERTa no fue el primer modelo en hacerlo, pero introdujo una variante de esta técnica llamada "disentangled attention", que le daba más flecibilidad al modelo en como podía combinar información semántica y posicional.

## Transfomers de solo decodificación
