---
tipo: concepto
categoria: machine-learning
subcategoria: supervised
tags:
  - arboles
  - supervised
  - id3
  - desorden
  - entropia
seccion_roadmap: Decision Trees
orden_seccion: 10
orden: 2
dificultad: media
---
## Aprendizaje: árboles de identificación, desorden

En el caso de que nuestra base de conocimiento sea una representación de información simbólica y no numérica, el método de vecinos más cercanos no es aplicable.

Esta información puede no ser importante para llegar a una conclusión o solo importar en ciertos casos. También es importante contemplar que el cálculo para realizar la clasificación puede ser costoso.

Para este tipo de información se contempla el marco de trabajo de pruebas, lo que lleva a componer un **árbol de pruebas** o **árbol de identificación**.

Para evaluar estos árboles se calcula el número de datasets homogéneos que generan las pruebas planteadas. La prueba con mayor coeficiente se aplica primero; si queda algún dataset heterogéneo, se vuelve a calcular la puntuación con la información restante.

Esto se relaciona con la **navaja de Ockham**: usualmente, la solución más sencilla tiende a ser la correcta.

[![Cálculo de desorden en árboles de identificación](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/raw/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020260303095611.png)](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/blob/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020260303095611.png)

Para grandes volúmenes de datos, es improbable separar los grupos en particiones homogéneas, así que hay que evaluar las pruebas de forma alternativa mediante una medida de desorden.

[![Medición del desorden](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/raw/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020260303094439.png)](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/blob/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020260303094439.png)

Esta herramienta nos permite medir el desorden para cada uno de los _subsets_, de modo que, si sumamos los diferentes valores de desorden de los subgrupos resultantes de la prueba, podremos darle un valor numérico para elegir entre unas pruebas u otras.

También hay que tener en cuenta la cantidad de muestras de cada grupo comparada con las muestras que han entrado en la prueba.

[![Factor de ponderación por tamaño de muestra](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/raw/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020260303094919.png)](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/blob/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020260303094919.png)

[![Ejemplo de árbol de decisión 1](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/raw/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020260303095307.png)](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/blob/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020260303095307.png)  
[![Ejemplo de árbol de decisión 2](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/raw/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020260303095519.png)](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/blob/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020260303095519.png)

Una vez vistos los resultados de los árboles anteriores, podemos concluir que, en ocasiones, se pueden simplificar las reglas sin afectar al resultado final de identificación.

En el ejemplo de reconocimiento de vampiros, podemos observar que toda la gente que come ajo no es vampiro, por lo que la prueba de la sombra es redundante y se puede obviar para hacer un sistema más simple.

---
