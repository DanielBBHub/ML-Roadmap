---
tipo: concepto
categoria: machine-learning
subcategoria: supervised
tags:
  - knn
  - instance-based
  - distancia
  - lazy-learning
seccion_roadmap: K-Nearest Neighbors (KNN)
orden_seccion: 9
orden: 1
dificultad: media
---
## Introducción al aprendizaje, vecinos cercanos

El aprendizaje puede darse de dos maneras distintas:

### 1. Aprendizaje basado en observaciones / regularidad

- Computación bulldozer
- Vecinos cercanos (_nearest neighbours_) — reconocimiento de patrones
- Redes neuronales — intento de imitar la biología
- Boosting — teoría

### 2. Restricciones / naturaleza humana

- Una única vez (_one shot_)
- Aprendizaje por explicación (_explanation based learning_)

### Vecinos cercanos


Este método se basa en la detección de características del objeto y la comparación con una base de conocimiento previamente adquirida, lo cual conduce a la respuesta de qué objeto es.

Ejemplos:

- Imagen de cuadrado → 4 lados, área completa → comparación de características → cuadrado
- Imagen de triángulo → 3 lados, área completa → comparación de características → triángulo
- Imagen de donut → 1 lado continuo, área con agujero → comparación de características → donut

La idea clave es que, una vez tenemos el muestreo original, debido a la variabilidad que puede introducir la entrada al sistema, las muestras no siempre caerán en el mismo punto del espacio donde está la referencia, pero sí cerca.

Un acercamiento posible es calcular la distancia de la muestra respecto a las diferentes muestras de referencia y asignarla a la más cercana. Otra posibilidad sería asignar de forma estática espacios para cada una de las muestras y separar directamente los resultados.

Con esta información se podrán distinguir las diferentes clases, separándolas por distancias a las referencias y agrupando todas las muestras en distintas categorías.

---
