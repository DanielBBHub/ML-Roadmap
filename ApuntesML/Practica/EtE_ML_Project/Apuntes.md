A la hora de plantear un proyecto de ML, la tarea principal es definir una lista de objetivos a cumplir para implementar una solución:

1. Define el problema y miralo con perspectiva
2. Obtén la información
3. Revisa la información
4. Procesa la información para exponer los patrones a los algoritmos de Machine Learning
4. Prueba varios modelos y lista los mejores
6. Haz fine tunning de los modelos y combínalos para mejorar la solución
7. Presenta la solución
8. Despliega, monitorea y mantén el sistema


## Mira con perspectiva

Lo primero que hay que tener en cuenta al empezar un proyecto es la información sobre la que vas a basar el resto de la implementación, en este ejemplo se ha escogido la información del censo de California 1990 del precio de la vivienda. Esta información tiene métricas como la población, el salario medio y el coste medio de la vivienda por bloque de la ciudad.

## Definir el problema
Hay que tener claro cual es el objetivo de implementar un modelo de ML, ¿de qué manera beneficia la solución implementada? La respuesta a esta pregunta moldeará como defines el problema, que algorítmos se escogerán, cuales serán las métricas de evaluación del modelo y cuanto trabajo se volcará en mejorar partes de este sistema

### Definición: *Pipelines*
```Las "pipelines" son una secuencia de componentes de procesado de datos. Estas son muy comunes en los sitemas de inteligencia artificial ya que hay grandes volúmenes de información para manipular y muchas transformaciones que aplicar.```

Por otro lado, es importante comprobar la solución actual al problema, si la hay, ya que nos aportará un sistema de referencia, tanto para el desempeño de la nueva solución así como pequeñas pistas sobre como resolver el problema.

Con esta información clara, es momento de empezar a diseñar el sistema. Primero tendremos que definir el problema, ¿La solución para por una implementación de un sistema de aprendizaje supervisado, no supervisado, con refuero? ¿Es una tarea de clasificación, regresión u otra cosa? ¿Deberemos utilizar aprendizaje por lotes o aprendizaje en linea?

En el caso del modelo de predicción de precios, podría implementarse una solución de aprendizaje supervisado, ya que la información viene etiquetada con la respuesta (el precio medio de la vivienda). Por otor lado, también es una tarea clásica de regresión con la que predecir el valor, en concreto, regresión multivariable con varias características.

## Elegir una métrica de evaluación para el modelo
En los problemas de regresión lineal una métrica típica es RMSE (raíz cuadrada de la media del error)

$$RMSE(X,h) = \sqrt{\frac{1}{m} \sum^m_{i=1} (h(x^i) - y^i)^2}$$
