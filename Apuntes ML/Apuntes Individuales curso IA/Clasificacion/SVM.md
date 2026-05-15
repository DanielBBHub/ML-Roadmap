---
tipo: concepto
categoria: machine-learning
subcategoria: supervised
tags:
  - clasificacion
  - supervised
  - kernel
  - margen-maximo
seccion_roadmap: Support Vector Machines
orden_seccion: 8
orden: 1
dificultad: avanzada
---
## Aprendizaje: Support Vector Machines

Las **Support Vector Machines (SVM)** son una técnica de aprendizaje supervisado basada en la construcción de una **frontera de decisión** que separe clases de forma óptima.

La idea principal no es solo separar las muestras, sino hacerlo con el **mayor margen posible** entre clases.

---

### 1. Frontera de decisión

Supongamos que queremos separar dos clases en un espacio vectorial.

La SVM define una frontera mediante un vector normal (W→), perpendicular a la línea o hiperplano separador, y una muestra W→ que queremos clasificar.

La regla de decisión básica es:

U→⋅W→≥C

o equivalentemente:

U→⋅W→+b≥0

Si esta condición se cumple, la muestra se clasifica como perteneciente a la clase positiva.

---

### 2. Restricciones de separación

Para que la separación sea clara, se imponen restricciones sobre las muestras de entrenamiento.

Para las muestras positivas:

W→⋅X+→+b≥1

Para las muestras negativas:

W→⋅X−→+b≤−1

Si definimos una etiqueta binaria (Yi) tal que:

- Yi=+1 para muestras positivas
- Yi=−1 para muestras negativas

entonces podemos unificar ambas expresiones en una sola:

Yi(Xi→⋅W→+b)≥1

Para las muestras que quedan exactamente sobre las fronteras del margen:

Yi(Xi→⋅W→+b)−1=0

---

### 3. El margen máximo

La SVM busca maximizar el **ancho del margen** entre las dos clases.

Si tomamos dos puntos límite del margen, (X+) y (X−), el ancho puede expresarse como:

ANCHO=(X+−X−)⋅W→|W|

Usando las restricciones anteriores, este ancho se simplifica a:

ANCHO=2|W|

Por tanto:

- maximizar el margen equivale a maximizar 2|W|
- lo cual equivale a minimizar |W|
- y, por conveniencia matemática, minimizar:

12|W|2

---

### 4. Optimización con multiplicadores de Lagrange

Para resolver el problema con restricciones se usa la formulación de **Lagrange**:

L=12|W|2−∑iαi[Yi(W→⋅Xi→+b)−1]

donde:

- αi son los multiplicadores de Lagrange,
- y las restricciones deben cumplirse.

---

### 5. Derivadas parciales

Para encontrar el extremo de la función, derivamos respecto a $\vec{W}$y respecto a b.

#### Respecto a (W→)

∂L∂W→=W→−∑iαiyixi→=0

De aquí se obtiene:

W→=∑iαiyixi→

Esto demuestra que el vector de decisión $(\vec{W})$es una **combinación lineal de las muestras**.

#### Respecto a b

∂L∂b=−∑iαiyi=0

Por tanto:

∑iαiyi=0

---

### 6. Forma dual del problema

Sustituyendo la expresión de $(\vec{W})$en la función original, el problema se transforma en una formulación dual:

L=∑iαi−12∑i∑jαiαjyiyjxi→⋅xj→

Este resultado es muy importante porque muestra que la optimización depende únicamente del **producto escalar entre pares de muestras**.

Esto significa que la información relevante para aprender la frontera está codificada en los productos:

xi→⋅xj→

---

### 7. Regla final de clasificación

Una vez encontrado (W→) la clasificación de una nueva muestra $(\vec{U})$se realiza con:

Si ∑iαiyixi→⋅u→+b≥0 entonces +

En caso contrario, la muestra se clasifica como negativa.

---

### 8. Ventajas de SVM

Las SVM tienen propiedades muy interesantes:

- no se quedan atrapadas en **máximos locales**
- buscan una solución óptima global
- solo dependen de los **vectores de soporte**
- maximizan el margen, lo que mejora la generalización

Los **vectores de soporte** son aquellas muestras que quedan más cerca del margen y que realmente determinan la posición del hiperplano separador.

---

### 9. Problema de la separabilidad lineal

La gran limitación de una SVM lineal es que solo funciona bien si las clases son **linealmente separables**.

Si las muestras no pueden separarse con una línea recta o hiperplano en el espacio original, la SVM lineal falla.

---

### 10. Transformación al espacio de características

Para resolver este problema se introduce una transformación del espacio original a otro espacio donde la separación sea posible:

ϕ(x→)

La idea es transformar las muestras a un espacio de mayor dimensión donde sí puedan separarse linealmente.

En ese nuevo espacio, el problema se convierte en uno de separación lineal sobre los vectores transformados:

ϕ(xi→)⋅ϕ(xj→)

y para clasificar una muestra nueva:

ϕ(xi→)⋅ϕ(u→)

---

### 11. Kernel trick

Aquí aparece la idea clave del **kernel**.

Si existe una función (K(x_i, x_j)) tal que:

K(xi,xj)=ϕ(xi→)⋅ϕ(xj→)

entonces podemos trabajar directamente con (K) sin calcular explícitamente la transformación (\phi).

Esto es el **kernel trick**.

### Ventaja

No hace falta construir el espacio transformado de forma explícita, pero aun así se obtiene el efecto de estar trabajando en él.

---

### Ejemplos de kernels

#### Kernel lineal

K(x,v)=x⋅v

#### Kernel polinómico

K(x,v)=(x⋅v+1)n

#### Kernel radial basis function (RBF)

K(x,v)=e−|x−v|22σ2

---

### 12. Importancia del kernel

El kernel permite resolver problemas no linealmente separables mediante una transformación implícita a un espacio más útil.

Esto hace que las SVM sean muy potentes para:

- clasificación
- reconocimiento de patrones
- separación de datos complejos
- problemas con fronteras no lineales

---

### 13. Limitaciones

Aunque son muy potentes, las SVM también tienen limitaciones:

- requieren elegir bien el kernel
- pueden sobreajustar si el kernel es demasiado flexible
- no son tan intuitivas de interpretar como otros modelos
- su rendimiento depende mucho de la parametrización

---

### 14. Idea final

Las **Support Vector Machines** buscan la frontera de decisión que **maximiza el margen** entre clases.

Su formulación matemática permite:

- convertir el problema en una optimización convexa,
- garantizar una solución global,
- y extender el método a casos no lineales mediante kernels.

En resumen:

> una SVM no solo separa clases, sino que intenta separarlas de la forma más robusta posible.
