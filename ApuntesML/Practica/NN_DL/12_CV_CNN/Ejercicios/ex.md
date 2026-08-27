# Ejercicios
## 1. ¿Cuáles son las ventajas de una CNN frente a una DNN totalmente conectada para la clasificación de imágenes?
- Las capas convolucionales estan completamente conectadas, lo cual hace que sean más rapidas de entrenar, reduce el riesgo de overfitting y necesita menos información para el entrenamiento
- Las entradas de los modelos son imágenes.  
- El funcionamiento de la convolución en imágenes permite captar patrones, tanto espaciales como de profundidad y una vez entrenada, puede captarlo en cualquier parte de la imagen 
- Las CNN pueden ser mixtas (varias capas convolucionales y una capa densa de salida) o completamente convolucionales  
- Utilizando el multi-output, las CNN pueden ser utilizadas tanto para clasificar como para localizar objetos de la imagen  
- Existen capas cuya función específica es reducir el tamaño de la imágen y aliviar el coste computacional del entrenamiento/inferencia

## 2. Considera una CNN compuesta por tres capas convolucionales, cada una con núcleos de 3 × 3, un stride de 2 y relleno ("padding") "same". La capa más baja produce 100 mapas de características, la intermedia produce 200 y la superior produce 400. Las imágenes de entrada son imágenes RGB de 200 × 300 píxeles:

### a. ¿Cuál es el número total de parámetros de la CNN?   
Teniendo en cuenta los filtros 3 x 3 y los 3 canales de la imágen de entrada y el término bias, 3 x 3 x 3 + 1 = 28, 28 parámetros por mapa de características. Sabiendo que la primera capa genera 100 mapas, eso sérian 2800  
De la misma manera, la segunda capa tiene un filtro de 3 x 3 y un término de bias, por los 100 mapas generados en la capa anterior 3 x 3 x 100 + 1 = 901, esto por los 200 mapas que genera esta capa, 901 x 200 = 180200  
Finalmente la tercera, también tiene un filtro 3 x 3 y un término bias por los mapas generados en la capa anterior 3 x 3 x 200 + 1 = 1801, teniendo en cuenta los 400 mapas que genera 1801 x 400 = 720400.  
Esto resulta en un total de 2800 + 180200 + 720400 = 903400 parámetros    

### b. Si usamos flotantes de 32 bits, ¿cuánta RAM requerirá como mínimo esta red al hacer una predicción para una sola instancia?    
Para calcular el tamaño de la imágen en cada punto, sabiendo que tenemos stride 2 y padding same, podemos afirmar que se partira en 2 en cada una de las capas, redondeando hacia arriba si es necesario. Dicho esto, los mapas de características serán de los siguientes tamaños:
- 100 x 150 en la primera capa
- 50 x 75 en la segunda capa
- 25 x 38 en la tercera capa

Sabiendo que 32 bits = 4 bytes en cada capa necesitaríamos el siguiente espacio:
- 4 x 100 x 150 x 100 = 6MB
- 4 x 50 x 75 x 200 = 3MB
- 4 x 25 x 38 x 400 = 1.5MB

Finalmente, y sabiendo que en las CNN solo se necesita tener en RAM los calculos de 2 capas simultaneamente, como máximo necesitaremos 6 + 3 = 9MB para las capas, además hay que tener en cuenta los 903400 parámetros en 4 bytes = 3.6MB, con lo que la RAM necesaria serían unos 9 + 3.6 = 12.6MB

### c. ¿Y qué ocurre al entrenar con un mini-lote de 50 imágenes?
Debido al back-prop, es necesario tener guardado todos los calculos hechos, con lo que al entrenar con un lote de 50 imágenes, habría que tener en cuenta todo el espacio en memória del modelo en una sola instancia y multiplicarlo x 50. Para cada instancia, las capas ocupan 6, 3 y 1.5 MB = 10.5 MB x 50 = 525MB. Por otro lado, la imágen de entrada necesitará 4 x 200 x 300 x 50 = 36MB y además el espacio para los parámetros del modelo, 3.6MB. En total se necesitarían 525 + 36 + 3.6 = 564.6 MB de RAM para entrenar un lote de 50 imágenes

## 3. Si a tu GPU se le agota la memoria mientras entrena una CNN, ¿qué cinco cosas podrías probar para resolver el problema?
- Reducir el tamaño de lote
- Aumentar el stride/ nº de strides
- Utilizar la CPU para almacenar algunos datos
- Quantizar/reducir la precisión del modelo 32 -> 16 bits
- Eliminar capas

## 4. ¿Por qué querrías añadir una capa de max pooling en lugar de una capa convolucional con el mismo stride?
Por que por el funcionamiento de max pooling, y para el caso de avg pooling, al utilizar un stride que reduzca las dimensiones de la imágen, este tipo de capas de agrupación pierden menos información, ya que tienen en cuenta los valores de los píxeles de toda la ventana para generar el nuevo valor de pixel. También te ahorra en espacio y en costo computacional, al no tener parámetros 

## 5. ¿Puedes nombrar las principales innovaciones de AlexNet en comparación con LeNet-5? ¿Y las principales innovaciones de GoogLeNet, ResNet, SENet, Xception, Efficient‐Net y ConvNeXt?

## 6. ¿Qué es una red totalmente convolucional? ¿Cómo se puede convertir una capa densa en una capa convolucional?
Una red totalmente convolucional es aquella que solo consta de capas convolucionales, de las más bajas a las más altas. Esta red generará una salida en forma de distribución de probabilidades sobre las clases entrenadas en el modelo.
Una capa densa puede transformarse en una convolucional tan fácilmente como utilizando los mismos pesos para la convolucional, solo hay que obtener los pesos resultados del entrenamiento y definirlos para la capa convolucional que la sustituya. Esta capa tiene que tener un filtro del tamaño de la capa de entrada, con un filtro por elemento de la anterior capa y utilizar padding valido.
## 7. ¿Cuál es la principal dificultad técnica de la segmentación semántica?
La principal dificultad de la segmentación semántica es el hecho de que la mayor parte de la información espacial se pierde a lo largo de la red convolucional, debido a la disminución de dimensiones de la imágen de entrada, con lo que acaba sabiendo que hay x objeto en un área indeterminada de la imágen original. Se suele utilizar skip-connections o agregar la imágen de entrada a traves de la red para mitigar este problema  

# Los siguientes ejercicios en un colab  


## 8. Construye tu propia CNN desde cero e intenta lograr la mayor precisión posible en MNIST.

## 9. Utiliza el aprendizaje por transferencia (transfer learning) para la clasificación de imágenes a gran escala, siguiendo estos pasos:

### a. Crea un conjunto de entrenamiento que contenga al menos 100 imágenes por clase. Por ejemplo, podrías clasificar tus propias fotos según la ubicación (playa, montaña, ciudad, etc.). Alternativamente, puedes usar un conjunto de datos existente, como el utilizado en el tutorial de PyTorch sobre transfer learning para visión por computadora.  
### b. Divídelo en un conjunto de entrenamiento, un conjunto de validación y un conjunto de prueba.  
### c. Construye el pipeline de entrada, aplica las operaciones de preprocesamiento adecuadas y, opcionalmente, añade aumento de datos (data augmentation).  
### d. Ajusta finamente (fine-tune) un modelo preentrenado con este conjunto de datos.

## 10. Realiza el tutorial de PyTorch sobre ajuste fino (fine-tuning) para detección de objetos. [ https://homl.info/detectiontuto ]