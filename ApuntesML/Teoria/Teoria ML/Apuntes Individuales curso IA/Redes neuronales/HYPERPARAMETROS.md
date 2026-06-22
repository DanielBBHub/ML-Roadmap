# Configuración de hyperparametros

Si bien las redes neuronales tienen una gran flexibilidad, esta es tambien una de sus mayores debilidades; hay demasiados hyperparámetros que configurar. No solo puedes utilizar una cantidad abismal de arquitecturas para implementar una red, si no que dentro de esta red puedes modificar el número de capas, de neuronas y sus funciones de activación, la lógica que inicializa los pesos de las neuronas, el optimizador, la tasa de aprendizaje, tamaño del batch ...

## Capas internas

Para muchos problemas puedes empezar con una única capa interna y obtener unos resultados tolerables, pero las redes profundas tienen una eficiencia de parametros mucho mas alta que aquellas redes "superficiales"; las redes profundas pueden modelar funciones complejas utilizando menos neuronas de manera exponencial que su contraparte.

Esto se da debido a que las diferentes capas permiten a la red neuronal componer características a traves de diferentes niveles. Por ejemplo: la primera capa en un clasificador de caras puede aprender a reconocer caracteristicas como puntos, arcos o lineas rectas, la segunda puede aprender a combinar las características anteriores y formar cuadrados o circulos y la tercera puede aprender a combinar las características nuevas para reconocer una boca, un ojo... y finalmente la última capa puede utilizar la información recibida para diferenciar entre caras.

Esta arquitectura jerárquica no solo ayude a las redes neuronals a converger más rápido hacía una solución, si no que mejora su habilidad de generalizar en nuevos conjuntos de datos. Es decir, si tu modelo ya se ha entrenado para reconocer caras en imágenes y quieres implementar una nueva para reconocer sus peinados, puedes utilizar las capas más bajas para empezar el entrenamiento y en vez de inicializar los pesos de manera aleatoria, utilizar los pesos que ya se han establecido en la primera red. A esto se le conoce como "Transfer Learning".

## Número de neuronas por capa interna

El número de neuronas en la entrada y la salida esta determinada por el tipo de entrada y la salida que requiere la tarea, en el ejemplo de clasificación de "Fashion-MNIST", la entrada requiere 2784 neuronas de entrada y 10 neuronas de salida (debido al tamaño de las imagenes reescaladas y las posibles clases).

Respecto a las capas internas, una práctica comun es diseñarlas con forma de pirámide, teniendo cada vez menos neuronas cuantas mas capas haya, la explicación detras de esta lógica es que las capas inferiores son capaces de combinar características que se reusaran en las capas siguientes.

De la misma manera que con las capas internas, se puede probar a aumentar las neuronas de cada capa hasta que se pueda apreciar overfitting en el modelo, o entrenar un modelo con mas capas y neuronas de las que necesitas y utilizar "early stopping" y otras tecnicas de regularización para prevenir la sobreespecialización.

En definitiva, tiene un impacto mayor utilizar más capas internas que añadir más neuronas a estas capas.

## Tasa de aprendizaje

La tasa de aprendizaje es un hiperparámetro altamente importante, normalmente el valor óptimo de este es, más o menos, la mitad de la tasa de aprendizaje máxima. 

Una manera práctica de encontrar un buen valor es entrenar el modelo por varios cientos de iteraciones, empezando con un valor pequeño (pj: 10⁻⁵) e incrementarlo gradualmente con las iteraciones. Ayuda a escoger un buen valor el hecho de plasmar gráficamente la pérdida como una función de la tasa de aprendizaje.

## Tamaño del lote

El tamaño del lote puede tener un impacto significativo en el desempeño del modelo y el tiempo de entrenamiento. La mayor ventaja de los aceleradores de HW, como las GPUs es que pueden procesar grandes lotes de manera eficiente, con lo que el algorítmo de entrenamiento puede ver mas instancias por segundo, con lo que algunos invesitgadores recomiendan utilizar el lote máximo que quepa en la VRAM. Esto puede llevar a problemas en los entrenamientos de modelos pequeños, resultando en una mala generalización de la información.

Dicho lo anterior, hay dos escuelas o métodos de definir el tamaño del lote:
1. Tamaño de lote grande, con "calentamiento" de la tasa de aprendizaje (tasa baja que va incrementando con las iteraciones)
2. Tamaño de lote pequeño

## Más hiperparámetros

### Optimizador

A veces utilizar otro optimizador en vez de gradiente de descenso con lotes pequeños puede acelerar el tiempo de entrenamiento y mejorar el rendimiento

### Función de activación

Normalmente la función ReLU es una buena función de activación para las capas intermedias, aun que puede ayudar reemplazarla con otras como tanh o la sigmoide.