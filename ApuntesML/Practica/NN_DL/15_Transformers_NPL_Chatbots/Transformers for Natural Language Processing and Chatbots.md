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