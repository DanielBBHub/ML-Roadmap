---
tipo: concepto
categoria: machine-learning
subcategoria: supervised
tags:
  - arboles
  - supervised
  - razonamiento
  - and-or
seccion_roadmap: Decision Trees
orden_seccion: 10
orden: 1
dificultad: media
---
## Reasoning: Goal Trees and Problem Solving

### Resolución de problemas del modelo

- **Generar tests**
- **Reducción del problema**
### Árbol de reducción del problema

### Transformaciones y ejemplo

### Reflexiones

---

Para abordar problemas complejos, como la siguiente integral:

[![Integral compleja](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/raw/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020251029134252.png)](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/blob/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020251029134252.png)

la estrategia consiste en **descomponerlo en problemas más sencillos**, un método conocido como **reducción de problemas**.

[![Descomposición del problema](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/raw/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020251029135000.png)](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/blob/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020251029135000.png)

Para resolver esta integral, se plantean una serie de teoremas que serán útiles tanto para este caso como para otros problemas de mayor complejidad.

Con el fin de verificar el proceso, se podría implementar un algoritmo con los siguientes pasos:

1. Aplicar los teoremas simples o seguros.
2. Consultar la tabla de integrales.
3. Verificar el resultado.

Dado que el tercer diagrama descompone la integral original en subintegrales, el **árbol de reducción del problema** se expande. Esto introduce un **nodo AND**, el cual requiere que la resolución de **todos** sus subproblemas sea exitosa para considerar resuelto el problema principal.

[![Árbol de reducción con nodo AND](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/raw/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020251029135548.png)](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/blob/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020251029135548.png)

Con el framework construido hasta ahora, basado en el planteamiento de teoremas, el programa aún no sería capaz de resolver la integral completa. Debido a esta limitación, será necesario incorporar un **algoritmo heurístico**. El objetivo es transformar el problema buscando equivalencias para **reducir o simplificar** su resolución.

Puede darse el caso de que existan **varias posibles soluciones** a nuestro problema, lo que nos llevaría a la adición de un **nodo OR**.

[![Árbol con nodos AND y OR](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/raw/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020251029152910.png)](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/blob/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020251029152910.png)

> Para diferenciar entre nodos AND y OR se utiliza el “cuarto de luna”, formando una figura parecida a la A.

En este caso, para decidir entre los nodos derivados del OR, se evalúa la **profundidad de la expresión matemática**. En esencia, se ha hecho una **evaluación heurística** de los nodos resultantes para discernir qué camino seguir en la resolución final.

[![Algoritmo completo de resolución](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/raw/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020251029154407.png)](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/blob/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020251029154407.png)

En definitiva, el programa aplica transformaciones conocidas (teoremas) y comprueba en la tabla.  
Si el problema está resuelto, se valida.  
Si no lo está, se evalúan los nodos con el objetivo de encontrar un siguiente problema derivado del original (ya sea AND u OR), se comprueba si es posible aplicar una transformada heurística y se cierra el bucle.

Con este procedimiento se logra resolver un problema complejo, reduciendo en apartados los diferentes pasos de la resolución y decidiendo qué caminos se escogerán para realizarla de forma más rápida.

### Catecismo inicial para desarrollar un programa así

- **Qué tipo** de conocimientos se necesitan
- **Cómo** se ha de representar ese conocimiento
- **Cómo** se ha de usar el conocimiento
- **Cuánto** conocimiento
- **Qué exactamente**

---


## Reasoning: Goal Trees and Rule-Based Expert Systems

[](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/blob/main/Apuntes%20ML/README.md#3-reasoning-goal-trees-and-rule-based-expert-systems)

En un ejemplo sencillo en el que un programa debe seguir un comando que ordena poner una caja encima de otra, se vuelve a generar un **árbol de decisión/objetivo**.

[![Árbol de decisión para apilar cajas](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/raw/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020251030160632.png)](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/blob/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020251030160632.png)

Este árbol contiene información que conforma una **traza**, es decir, información sobre el planteamiento y razonamiento de la solución.

Se pueden responder preguntas de:

- **“¿Por qué haces X?”** mirando hacia arriba en la traza.
- **“¿Cómo haces X?”** bajando hacia abajo en la traza.

[![Metáfora de la hormiga de Simon](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/raw/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020251031081914.png)](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/blob/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020251031081914.png)

La figura de arriba representa el camino de una hormiga en una playa, descrito en la **metáfora de la hormiga de Simon**.

Esta metáfora quiere representar que la complejidad del comportamiento de un programa viene dada por el máximo entre la complejidad del programa y la complejidad del problema a resolver:

max(Ccomportamiento)=max(Cprograma,Cproblema)

---

### Sistemas basados en reglas

[](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/blob/main/Apuntes%20ML/README.md#sistemas-basados-en-reglas)

Los **sistemas basados en reglas** utilizan conocimiento encapsulado en reglas simples con las que realizan asociaciones para llegar a conclusiones.

[![Sistema basado en reglas - identificación de animales](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/raw/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020251031082939.png)](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/blob/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020251031082939.png)

En este caso, un sistema basado en reglas utiliza los hechos presentados para enlazarlos de forma progresiva con el objetivo de llegar a la conclusión de qué animal se está hablando.  
Este sistema se considera **Forward-Chaining**.

También se puede revertir el proceso y trabajar a partir de una hipótesis, por ejemplo: “es un guepardo”, para encontrar las evidencias de que lo es.

Al conjunto de estas modalidades podríamos considerarlo un sistema de deducción, en modo **hacia delante / hacia atrás**.

---

## Search: Depth-First, Hill Climbing, Beam


Otro concepto importante es el uso de **la lista de la cola**, es decir, los nodos que vas a visitar o has visitado. Esto ayuda a definir el orden de ejecución y a llevar la cuenta de los nodos ya explorados.

Esto mejora el rendimiento del algoritmo, porque al saber que ya hemos visitado un nodo y que hemos extendido el camino a partir de él, podemos evitar volver a evaluarlo. Esto es aplicable a **Depth-First** y **Breadth-First**.

### Hill Climbing


Existe otro método de búsqueda llamado **Hill Climbing**, que ignora el orden léxico de los nodos y los ordena por su cercanía a la meta.

[![Algoritmo Hill Climbing](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/raw/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020251104185222.png)](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/blob/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020251104185222.png)

En este caso, la unión entre los nodos tenía asignado un valor arbitrario, utilizado para elegir el camino más corto, siendo este el resultado. Esto es otro ejemplo de **evaluación heurística** para agilizar la resolución del problema.

next(node)=min(length(NODEn))

### Beam Search


Por último, existe la búsqueda **BEAM**, una variante de **Breadth-First** con un número constante de niveles tras los cuales se aplica una evaluación heurística.

[![Algoritmo BEAM Search](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/raw/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020251104185929.png)](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/blob/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020251104185929.png)

En este ejemplo se aplica una profundidad máxima de 2 nodos, tras la cual se evalúa cuál de los dos nodos está más cerca del objetivo.

### Algoritmo general de búsqueda


[![Algoritmo general de búsqueda](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/raw/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020251104190258.png)](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/blob/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020251104190258.png)

El concepto es crear una cola y extender sobre los nodos elegidos. Dependiendo del tipo de búsqueda, tendrán estas propiedades:

- **Depth-First**: se encolará al principio de la cola.
- **Breadth-First**: se encolará al final de la cola.
- **Hill Climbing**: se encolará al principio de forma ordenada según una evaluación heurística.
- **Beam**: se encolará al principio de la cola en base a la mejor W.

---

## Minimax, Alpha-Beta


Cuando se genera un árbol de decisión en el que intervienen dos actores, cada nivel de nodos representa la acción de uno de ellos.  
El nodo inicial representa el estado del primer actor, así como los nodos impares, mientras que el segundo nodo, así como los nodos pares, representan los estados del segundo actor.

A este actor principal se le llama **Max**, mientras que al actor secundario se le denomina **Min**. Esta es la base del algoritmo **Minimax**.

Con estas denominaciones se pretende que el actor Max elija los nodos con mayor valor, mientras que Min haga lo contrario.

[![Algoritmo Minimax](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/raw/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020251214095818.png)](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/blob/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020251214095818.png)

En este ejemplo, el jugador Min escogería el nodo con valor 2, el mayor de sus opciones, dejando al jugador Max con las opciones 2 y 7, de las cuales elegiría 7.

El número de posibles nodos finales es igual al ancho del árbol (**b**) elevado a la profundidad de búsqueda (**d**):

bd

### Alpha-Beta pruning

Desarrollando el algoritmo Minimax, aparece **Alpha-Beta** para remediar la necesidad de potencia de procesado que requiere.  
El principal problema del algoritmo Minimax es que su objetivo óptimo es llegar lo más lejos posible para tomar la mejor decisión, lo cual genera un gran número de posibilidades.

El algoritmo **Alpha-Beta** intenta solucionar este problema eliminando ramas enteras basándose en sus valores estáticos. Se evalúan los nodos finales y se les adjudican valores a los nodos previos.

El resultado no es ni el mayor ni el menor número, sino el mejor número dentro de un camino en el que se han hecho compromisos para maximizar los valores del actor Max mientras se minimizan los valores para el actor Min.

[![Algoritmo Alpha-Beta](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/raw/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020251214102821.png)](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/blob/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020251214102821.png)

---
