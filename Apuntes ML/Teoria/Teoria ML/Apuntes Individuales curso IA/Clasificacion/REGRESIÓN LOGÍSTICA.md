---
tipo: concepto
categoria: machine-learning
subcategoria: supervised
tags:
  - clasificacion
  - supervised
  - probabilistico
  - bayes
seccion_roadmap: Logistic Regression
orden_seccion: 7
orden: 1
dificultad: media
---
## Inferencia probabilística I

La inferencia probabilística se basa en el concepto de **contar ocurrencias** de combinaciones posibles y convertir esos recuentos en probabilidades mediante normalización: nºcnºt

donde:

- nco = número de casos favorables
- nto = número total de casos

### Problema del enfoque tabular

El principal problema de este enfoque es que el número de combinaciones crece muy rápidamente con el número de variables.

Si tenemos n variables binarias, el número de filas en la tabla conjunta es:$$\large 2^n$$ Por tanto, el volumen de datos necesario crece exponencialmente.

### Interpretaciones de la probabilidad

Hay varias maneras de interpretar estos valores:

1. **Interpretación frecuentista**
    
    - la probabilidad representa la frecuencia con la que ocurre un evento en muchas repeticiones
2. **Interpretación subjetiva o bayesiana**
    
    - la probabilidad representa un grado de creencia
3. **Propensión natural**
    
    - la probabilidad se interpreta como una tendencia o disposición objetiva del mundo

### Motivación

Como las tablas conjuntas completas se vuelven rápidamente inmanejables, necesitamos **modelos matemáticos** que permitan representar y razonar con probabilidades de forma más compacta.

#### Probabilidad básica

Axiomas:

1. La probabilidad está acotada entre 0 y 1: 1≥P(a)≥0
2. Probabilidad binaria
    1. P(verdadero)=1
    2. P(falso)=0
3. Regla de adición: P(a)+P(b)−P(a,b)=P(a|b)

#### Probabilidad condicional

Definición: P(a|b)≡P(a,b)P(b) Por lo que la probabilidad siguiente: P(a,b,c) es igual a: P(a,b,c)=P(a,y) teniendo en cuenta que y=b,c P(a,y)=P(a|y)P(y)=P(a|b,c)P(b,c)=P(a|b,c)P(b|c)P(c)

#### Regla de la cadena

Lo cual puede generalizarse de la siguiente manera:$$\large P(x_1....,x_n) = \prod_{i=n}^{1} P(x_i|x_{i-1}, ... x_i)$$

### Independencia
#### Definición

Dos variables a y b son independientes si: P(a|b)=P(a) si a es independiente de b

### Independencia condicional
#### Definición
a y b son condicionalmente independientes dado z si: P(a|bz)=P(a|z) De la misma manera$$\large P(ab|z) = P(a|z)P(b|z) $$Esto significa que, una vez conocido z , la información adicional sobre b ya no cambia la probabilidad de a .

#### Regla de la cadena
### Redes de creencias

Una **red de creencias** (_belief network_ o _Bayesian network_) es una representación compacta de una distribución conjunta usando:

- un **grafo dirigido acíclico**
- relaciones de dependencia entre variables
- tablas de probabilidad condicional para cada nodo

La idea clave es que **cada nodo solo depende de sus padres**.

En este ejemplo:

- **Perro** depende de **Ladrón** y **Ratón**
- **Policía** depende de **Perro**
- **Basura** depende de **Ratón**
- **Ladrón** y **Ratón** son variables independientes de raíz

### Reducción del número de parámetros
Si quisiéramos describir la distribución conjunta de 5 variables binarias con una tabla completa, necesitaríamos:

25=32

valores.

Pero usando una red bayesiana, basta con especificar las probabilidades locales:

- P ( Ladrón )
- P ( Ratón )
- P ( Perro ∣ Ladrón , Ratón )
- P ( Policía ∣ Perro )
- P ( Basura ∣ Ratón )

Esto reduce drásticamente el número de parámetros.

Teniendo las siguientes probabilidades: P(Ladron)=0.1,P(Raton)=0.5

La probabilidad de que el perro ladre, se da en la siguiente tabla:

|Policía|Ratón|P(Perro)|
|---|---|---|
|F|F|0.1|
|T|F|1.0|
|F|T|0.5|
|T|T|1.0|
|De la misma manera que la probabilidad de que se llame a la policía es la siguiente|||

|Perro|P(Policía)|
|---|---|
|F|0.01|
|T|0.1|
|Y las probabilidades de que la rata tire la basura las siguientes||

|Rata|P(Basura)|
|---|---|
|F|0.001|
|T|0.8|
|Gracias a la interdependencia de los nodos y sus probabilidades, se han reducido las variables necesarias de 2n (siendo n = 5) →10, las cuales se expresan en la siguiente formula de probabilidad dependiente: $$\large P(policia,perro,ladron,basura,raton) =||
|$$$$P(policia|perro,ladron,basura,raton)P(perro|
|Pudiendo eliminar de las probabilidades condicionales aquellas variables las cuales no tienen relación de dependencia.||

### Idea clave de la clase

La gran ventaja de las redes de creencias es que permiten:

- representar distribuciones complejas con muchos menos parámetros,
- aprovechar independencia condicional,
- calcular probabilidades conjuntas a partir de tablas locales pequeñas.

En el ejemplo de 5 variables:

- tabla completa: **32 valores**
- red bayesiana: **10 valores**

---

### Resumen

1. La probabilidad puede obtenerse por **frecuencias**.
2. Las tablas conjuntas crecen como 2 n .
3. La probabilidad básica se apoya en axiomas simples.
4. La probabilidad condicional permite descomponer distribuciones.
5. La independencia y la independencia condicional reducen la complejidad.
6. Las **redes de creencias** factorizaron el problema de forma eficiente.

## Inferencia probabilística II

En esta clase se profundiza en el uso de **redes bayesianas** para:

- calcular probabilidades conjuntas a partir de tablas locales,
- hacer **inferencia ingenua de Bayes**,
- y llegar finalmente al **descubrimiento de estructura**.

### Repaso: red bayesiana e inferencia

Podemos representar las relaciones entre nodos mediante una lista ordenada de variables, por ejemplo: $$C \text{ / } D \text{ / }B \text{ / }T \text{ / }R $$A partir de esa ordenación, la **regla de la cadena** permite escribir la distribución conjunta como: P(C,D,B,T,R)= $$P(C|D,B,T,R)P(D|B,T,R)P(B|TR)P(T|R)P(R) $$Sin embargo, si el grafo está bien definido, podemos usar la **independencia condicional** para simplificarla.

En una red bayesiana, cada variable es independiente de sus **no descendientes**, dado sus **padres**.  
Por tanto, la expresión anterior se reduce a:$$P(C|D)P(D|B,R)P(T|R)P(B)P(R) $$ Esto permite calcular cualquier combinación de valores a partir de tablas locales mucho más pequeñas. ![[Pasted image 20260416085757.png]]

### Modelo

Una vez obtenidas frecuencias de observación, podemos construir tablas de **probabilidades condicionales** para cada nodo.

Estas tablas son útiles porque, una vez fijados los valores de los nodos padres —por ejemplo, mediante una simulación o un sorteo—, solo hay que consultar la probabilidad correspondiente a esa combinación concreta. [![Pasted image 20260416085805](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/raw/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020260416085805.png)](https://github.com/DanielBBHub/MIT-ML-COURSEWARE/blob/main/Apuntes%20ML/fotos%20ml/Pasted%20image%2020260416085805.png)

### Idea clave
- Los **padres** determinan qué fila de la tabla usar.
- El nodo hijo se calcula a partir de esa probabilidad condicional.

El problema de este método es que depende críticamente de que las relaciones entre nodos estén bien definidas.

### Inferencia Naive Bayesiana

Partimos de la definición de probabilidad condicional:$$P(a|b) \equiv \frac{P(a,b)}{P(b)}$$ Podemos expresar P(a,b) indistintamente de las siguientes maneras:

- P(a,b)=P(a|b)P(b)
- P(a,b)=P(b|a)P(a)

Con lo que podemos reescribir la definición dela siguiente manera: P(a|b)=P(b|a)P(a)P(b)

### Interpretación en clasificación

Esta forma es especialmente útil en problemas de **diagnóstico** o **clasificación**.

- a = **clase** o causa que queremos predecir
    - por ejemplo: una enfermedad
- b = **evidencia** o síntomas observados

En muchos casos es más fácil modelar:

P(b∣a)

que modelar directamente:

P(a∣b)

Es decir:

- predecir los **síntomas dada la enfermedad** suele ser más fácil,
- que predecir la **enfermedad dados los síntomas**.

### Inferencia ingenua con evidencias independientes

Si la evidencia está formada por varias observaciones:

e1,e2,…,en

y suponemos que son **independientes dado la clase** ci , entonces:

P(e1,…,en∣ci)=∏j=1nP(ej∣ci)

Por tanto:

P(ci∣e1,…,en)∝P(ci)∏j=1nP(ej∣ci)

Esto es el principio del **Naive Bayes** o **clasificador ingenuo de Bayes**.

### Idea

La independencia condicional simplifica muchísimo el cálculo y permite hacer clasificación eficaz incluso con muchos síntomas o indicadores.

---

### Ejemplo: clasificación de monedas

Un ejemplo clásico es distinguir entre dos monedas:

- una moneda justa:$P ( H ) = 0.5$
- una moneda sesgada: P(H)=0.8

Supongamos que escogemos una moneda al azar y observamos la secuencia de lanzamientos.

### Paso 1: evidencia inicial

Si obtenemos:

- cara
- cruz

podemos comparar cuál de las dos monedas hace más probable esa secuencia.

### Interpretación

A medida que acumulamos evidencia, actualizamos la probabilidad de que la moneda elegida sea una u otra.

- si aparecen muchos resultados compatibles con la moneda sesgada, su probabilidad sube,
- si aparece un resultado incompatible, su probabilidad puede caer a 0 en algunos modelos.

Esto ilustra cómo la evidencia empuja la inferencia hacia una hipótesis u otra.

### Selección de modelos y descubrimiento de estructura

Aquí aparece una idea más potente: no solo queremos clasificar clases, sino también decidir **qué estructura de modelo es mejor**.

### Dos modelos posibles

Podemos tener dos redes o dos hipótesis estructurales distintas:

- modelo izquierdo
- modelo derecho

A partir de datos observados, comparamos cuál explica mejor las observaciones.

### Bayes para comparar modelos

Para cada modelo Mi :

P(Mi∣datos)∝P(datos∣Mi)P(Mi)

Si asumimos priors similares, la comparación puede hacerse principalmente por la verosimilitud de los datos bajo cada modelo.

---

### Descubrimiento de estructura

Esto va más allá de la clasificación: ya no solo se predice una clase, sino que se intenta descubrir **la estructura correcta del grafo**.

### Problema

Hay muchísimas redes posibles.

Incluso restringiendo:

- que no haya ciclos,
- y que cada nodo tenga pocos padres,

el número de estructuras posibles es enorme.

### Solución: búsqueda

No se prueban todas las redes.  
En su lugar, se hace **búsqueda** en el espacio de modelos:

1. se parte de una red inicial,
2. se modifica ligeramente,
3. se evalúa,
4. se mantiene la mejor encontrada,
5. y se repite.

---

### Función de puntuación

Para evitar problemas numéricos al multiplicar probabilidades muy pequeñas, se usa la suma de logaritmos:

∑log⁡P(modelo)

En lugar de multiplicar probabilidades, se suman sus logaritmos.

Esto evita el **underflow** y es más práctico computacionalmente.

---

### Máximos locales y reinicio aleatorio

El espacio de modelos suele ser:

- muy grande,
- muy irregular,
- lleno de máximos locales.

Por eso se usa:

- **búsqueda local**
- **reinicio aleatorio** (_random restart_)

La idea es:

- conservar el mejor modelo visto hasta el momento,
- pero de vez en cuando saltar a una región distinta del espacio de búsqueda,
- para evitar quedar atrapado en un máximo local.

---

### Aplicaciones

La clase termina destacando que la inferencia bayesiana es útil en muchos problemas de diagnóstico:

- diagnóstico médico,
- detección de mentiras,
- evaluación del conocimiento en exámenes,
- diagnóstico de fallos en sistemas,
- análisis de causas probables en programas o máquinas.

En todos estos casos:

- observamos **síntomas** o **evidencias**,
- y queremos inferir la **causa** más probable.

---

### Ideas clave de la clase

1. La regla de la cadena permite factorizar una distribución conjunta.
2. Las redes bayesianas aprovechan la independencia condicional.
3. Bayes permite invertir una relación causal o diagnóstica.
4. Naive Bayes simplifica el cálculo suponiendo independencia entre evidencias dadas la clase.
5. La inferencia bayesiana también sirve para comparar modelos.
6. El descubrimiento de estructura requiere búsqueda en un espacio enorme de hipótesis.
7. Se usa la suma de log-probabilidades para evitar problemas numéricos.
8. Los reinicios aleatorios ayudan a escapar de máximos locales.

---

### Resumen corto

La gran idea de esta clase es que la probabilidad no solo sirve para calcular resultados, sino también para:

- **clasificar clases**
- **comparar modelos**
- **descubrir estructuras**.
