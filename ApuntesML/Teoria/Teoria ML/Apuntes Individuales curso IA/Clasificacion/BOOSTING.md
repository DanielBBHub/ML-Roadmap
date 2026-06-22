---
tipo: concepto
categoria: machine-learning
subcategoria: supervised
tags:
  - ensemble
  - supervised
  - boosting
  - secuencial
seccion_roadmap: Gradient Boosting Machines
orden_seccion: 11
orden: 1
dificultad: avanzada
---
## 17. Aprendizaje: Boosting

### Idea 1: combinar clasificadores débiles

En el caso de un clasificador binario, cuya salida es `{-1, +1}`, si su tasa de error es demasiado alta, podemos combinar varios modelos para tomar una decisión conjunta basada en una “multitud”.

Si tenemos varios clasificadores `h_1`, `h_2`, `h_3`, podemos definir un clasificador final `H` como:

The following macros are not allowed: operatorname

  

$$
H(x)=\operatorname{sign}(h_1(x)+h_2(x)+h_3(x))
$$

La idea es que un clasificador fuerte puede construirse a partir de varios clasificadores débiles, siempre que sus errores no coincidan demasiado.

Para reducir el solapamiento entre errores, podemos usar un esquema secuencial:

- La información de entrada →h1
- La información con más peso en los fallos de h1→h2
- Las muestras donde h1 y h2 discrepan →h3`y

---

### Idea 2: estructura en árbol

Esta solución puede verse como una estructura en árbol:

No tiene por qué ser estática, ya que cada clasificador descendiente también puede tener sus propios clasificadores previos:

### Idea 3: decision tree stumps y error ponderado

Si tuviésemos un espacio 2D, un `decision tree stump` no sería un árbol completo, sino una única prueba.

Cada stump corresponde a una posible separación del espacio. El error de una prueba, si todas las muestras tienen el mismo peso, puede expresarse como: E=∑e1N donde la suma se hace sobre las muestras mal clasificadas.

Si los pesos de las muestras son distintos, entonces el error pasa a ser: $$E = \sum_e W_i$$con la condición: ∑Wi=1

Esto permite dar más peso a las muestras difíciles o “exageradas”.

### Idea 4: combinación ponderada de clasificadores

El clasificador final puede generalizarse asignando pesos a los distintos clasificadores:$$H(x) = \operatorname{signo}(\alpha_1_h_1 + \alpha_2_h_2 + \alpha_3_h_3 +...\alpha_n_h_n)$$Esto representa una especie de “sabiduría ponderada de una multitud de expertos”, donde cada clasificador aporta según su desempeño.

### Idea 5: algoritmo iterativo de boosting

A partir de esta idea, se puede construir un proceso iterativo que ajuste los pesos de los clasificadores y de las muestras con el objetivo de minimizar el error.

La lógica del algoritmo es:

1. Inicializar los pesos de las muestras.
2. Elegir el clasificador `h^t` que minimiza el error ponderado.
3. Calcular el peso `α^t` del clasificador.
4. Actualizar los pesos de las muestras.
5. Repetir el proceso.

### Idea 6: actualización de pesos

La actualización de los pesos puede escribirse como:$$w^{t+1}_i = \frac{w^{t}_i}{Z}e^{-\alpha^th^t(x_i)y_i}$$

donde:

- `Z` es un factor normalizador
- ht(xi) es la predicción del clasificador en la iteración t
- yi es la etiqueta correcta

El valor óptimo de αt viene dado por:$$\alpha^t = \frac{1}{2}\ln(\frac{1-\epsilon^t}{\epsilon^t})$$

donde ϵt es el error ponderado en la iteración t.

---

#### Observación importante
La actualización de pesos hace que:

- las muestras bien clasificadas reduzcan su peso relativo,
- las muestras mal clasificadas aumenten su importancia,
- y el siguiente clasificador se enfoque en corregir errores anteriores.
