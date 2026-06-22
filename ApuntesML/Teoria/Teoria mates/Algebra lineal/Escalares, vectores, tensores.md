---
tipo: concepto
categoria: matematicas
subcategoria: algebra-lineal
tags:
  - escalares
  - vectores
  - tensores
  - fundamentos
seccion_roadmap: Mathematical Foundations
orden_seccion: 1
orden: 1
dificultad: basica
---
## Escalares
Los escalares son números reales $\Re$ o complejos que tienen las siguientes características:
- Dimensión 0
- Se pueden representar como: $\lambda$, $\alpha$ 
- Se pueden sumar, restar, multiplicar y dividir

## Vectores
Un vector es una lista ordenada de escalares que representa un punto en el espacio o una dirección. Tienen las siguientes características:
- Dimensión 1
- Se suelen representar como: $\vec{v}$; $\vec{v} = (\lambda_1, ..., \lambda_n)$
- Existen en el espacio $\Re^n$

## Tensores
un tensor es la generalización de escalares, vectores y matrices de n- dimensión. Estos tienen:
- Orden/rango: número de dimensiones
- Dimensión/forma: arreglo de la información
- Se puede operar en cualquier eje

Dado un tensor 3D que representa las transformaciones posibles de un cubo$$T = \begin{bmatrix} \alpha_{11} && \alpha_{12} && \alpha_{13} \\
\alpha_{21} && \alpha_{22} && \alpha_{23} \\
\alpha_{31} && \alpha_{32} && \alpha_{33} \\\end{bmatrix}$$
Este tensor tiene 3 dimensiones, ya que representa un objeto 3D y cada fila/columna corresponde a una dimensión en concreto ($X,Y,Z$)

Por otro lado, y para ser mas precisos, el rango de un tensor es la cantidad de índices que son necesarios para encontrar un elemento en el tensor, en este caso, fila y columna; rango-2 $\alpha_{31}$.
