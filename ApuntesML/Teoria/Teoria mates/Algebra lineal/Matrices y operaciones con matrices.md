---
tipo: concepto
categoria: matematicas
subcategoria: algebra-lineal
tags:
  - matrices
  - operaciones-matriciales
  - suma
  - multiplicacion
  - transpuesta
seccion_roadmap: Mathematical Foundations
orden_seccion: 1
orden: 2
dificultad: basica
---
## Ecuaciones lineales

Muchos problemas se pueden representar mediante sistemas de ecuaciones y resueltos por álgebra lineal. Por ejemplo:
	Una compañía produce $N_1, ..., N_n$ productos con los recursos $R_1, ..., R_m$. Para producir un producto $N_j$ se necesitan $a_{ij}$ unidades de recurso $R_i$. Si producimos $x_1, ..., x_n$ unidades de los correspondientes productos, necesitariamos: $$a_{1j}x_i +...+a_{in}x_n$$recursos de $R_i$. Un plan óptimo tiene que satisfacer que: $$a_{11}x_1 + ... + a_{1n}x_n = b_1$$
	                            $a_{m1}x_1 + ... + a_{mn}x_n = b_m$
	donde $a_{ij} \text{ } \in \text{ } \Re$ y $b_{i} \text{ } \in \text{ } \Re$ 
Por otro lado, hay una interpretación geométrica de los sistemas de ecuaciones. En un sistema con $(x_1,x_2)$, los valores y la solución al sistema se pueden ver representadas en el plano definido por $(x_1,x_2)$. Esta solución puede ser una línea (si ambas ecuaciones describen la misma línea), un punto donde intersectan o un vacio (si son líneas paralelas).

En caso de tener tres variables, estas ecuaciones determinarán un plano y cuando hagamos la interseción deestos, la solución puede ser un plano, una línea, un punto o vacio.

## Matrices

Estas pueden utilizarse para representar de manera compacta sistemas de ecuaciones lineales, así como funciones lineales

### Definición
Siendo $(m,n) \in \Re$, la matriz A es una tupla de $m*n$ elementos, donde $a_{ij}, i=1,...,m; j=1,...,n$siendo $i$ y $j$ las filas y columnas de la matriz.

Las matrices $(1,n)$ y $(m,1)$ son llamadas matrices de línea/columna o vectores linea/columna.

### Suma
La suma de dos matrices $A \in \Re^{m*n}$ y $B \in \Re^{m*n}$ se define como la suma de sus elementos $$ A = \begin{bmatrix}  
1 & 2 & 3\\  
a & b & c  
\end{bmatrix}, B = \begin{bmatrix}  
4 & 5 & 6\\  
d & e & f  
\end{bmatrix} \rightarrow A+B = \begin{bmatrix}  
1+4 & 2+5 & 3+6\\  
a+d & b+e & c+f  
\end{bmatrix}$$

### Producto
El producto de matrices $A \in \Re^{m*n}$ y $B \in \Re^{m*n}$ se calcula como $C=AB \in \Re^{m*k}$ como: $$c_{ij} =  \sum_{l=1}^{n} a_{il}b_{lj}; i=1,...,m;j=1,...,k$$
Es decir, se mutiplican los elementos de la fila $\large i$ por los elementos de la columna $\large j$.

Las matrices solo pueden multiplicarse si las dimensiones vecinas son iguales, es decir, dada la matriz $A \in \Re^{n*k}$ y $B \in \Re^{n*k}$:
$\large A*B = C$ ✅
$\large \xcancel{B*A  C}$ ❌

### Identidad
La matriz identidad es aquella que tiene una línea diagonal de valores 1 y el resto 0

### Propiedades
- #### Asociativa
	- $V\llap{-}A \in \Re^{m*n},B \in \Re^{n*p},C \in \Re^{p*q} : (AB)C = A(BC)$  
- #### Distributiva
	- $V\llap{-}A,B \in \Re^{m*n},C,D \in \Re^{n*p} : (A+B)C = AC + BC; A(C+D) = AC+AD$
- #### Producto con la identidad
	- $V\llap{-}A \in \Re^{m*n}:I_mA=AI_n=A$

### Inversa y traspuesta
#### Definición
Dada la amtriz cuadrada $A \in \Re^{n*n}$ y $B \in \Re^{n*n}$, teniendo la propiedad de $AB=I_N=BA$, $B$ es la inversa de $A$ y se define como $A^-1$. Considerando la siguiente matriz: $$A:= \begin{bmatrix}  
a_{11} &a_{12}\\  
a_{21} & a_{22}   
\end{bmatrix} \in \Re^{2*2}$$
Si multiplicamos A con: 
$$A\prime := \begin{bmatrix}  
a_{22} & -a_{12}\\  
-a_{21} & a_{11}   
\end{bmatrix}$$

Obtenemos:
$$A*A\prime = \begin{bmatrix}  
a_{11}a_{22}- a_{12}a_{21}  & 0\\  
 0 & a_{11}a_{22}-a_{21}a_{11}  
\end{bmatrix}$$
Por lo que: 
$$A^-1 = \frac{1}{a_{11}a_{22} - a_{12}a_{21}}*\begin{bmatrix}  
a_{22} & -a_{12}\\  
-a_{21} & a_{11}   
\end{bmatrix}$$

Si y solo si $a_{11}a_{22} - a_{12}a_{21} \neq 0$

Para las matrices $A\in\Re^{m*n}$ y $B\in\Re^{mn*m}$ con $b_{ij} = a_{ji}$, se llama a $B$ la matriz traspuesta de $A$; $B=A^T$.

Normalmente, $A^T$ se consige utilizando las filas de $A$ como columnas de $A^T$.

#### Propiedades
- $A*A^-1 =I=A^-1*A$
- $(A*B)^-1=B^-1*A^-1$
- $(A+B)^-1\neq B^-1+A^-1$
- $(A^T)^T=A$
- $(A*B)^T = B^T*A^T$
- $(A+B)^T = A^T+B^T$

### Simetria
una matriz $A\in\Re{n*n}$ es simetrica si $A=A^T$. Solo pueden ser simétricas aquellas matrices que sean cuadradas; matrices$(n,n)$.

Si una matriz simétricfa es invertible, tambien lo es su $A^T$$$(A^-1)^T=(A^T)^-1=:A^-T$$
### Producto por un escalar

Dada una matriz $A \in \Re^{m*n}$ y $\lambda \in \Re$. Entonces, $\lambda A = K, K_{ij}= \lambda a_{ij}$. Practicamente $\Lambda$ escala cada elemento de $A$. Para $\lambda,\psi \in \Re$, se mantienen las propiedades:
- Asociativa
	- $(\lambda\psi)C=\lambda(\psi C), C\in\Re^{m*n}$
	- $\lambda (BC) = (\lambda B)C = B(\lambda C)=(BC)\lambda$
- Distributiva
	- $(\lambda + \psi)C = \lambda C + \psi C$
	- $\lambda (B + C) = \lambda B + \lambda C$

### Representacion de sistemas de ecuaciones lineales
Si consideramos:$$
\begin{cases}
  2x_1 + 3x_2 + 5x_3 = 1 \\
  4x_1 - 2x_2 - 7x_3 = 8 \\
  9x_1 + 5x_2 - 3x_3 = 2
\end{cases}
$$

Aplicando las reglas del producto de matrices, obtenemos lo siguiente:$$\begin{bmatrix}  
2 & 3 & 5\\  
4 & -2 & -7\\
9 & -5 & -3\\
\end{bmatrix}*
\begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} = 
\begin{bmatrix} 1 \\ 8 \\ 2 \end{bmatrix}$$
siendo $x_1, x_2, x_3$ escalares para la primera, segunda y tercera fila

### Resolución de sistemas de ecuaciones lineales
Dada la siguiente matriz de ejemplo:$$
\begin{bmatrix}1&0&8&-4\\0&1&2&12\\\end{bmatrix} *
\begin{bmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \end{bmatrix} = 
\begin{bmatrix} 42 \\ 8 \end{bmatrix}
$$

Teniendo en cuenta que tenemos que encontrar los valores escalares de tal manera que:$$\sum^4_{i=1}c_1x_1=b$$
donde $c_1$ es son los valores de la i-ésima columna y b la solución a la derecha, una solución del sistema puede ser la siguiente:$$b = 42*\begin{bmatrix} 1 \\ 0\end{bmatrix} + 8*\begin{bmatrix} 0 \\ 1\end{bmatrix} = \begin{bmatrix} 42 \\ 8\end{bmatrix}$$

es decir $\begin{bmatrix} 42 & 8 & 0 & 0\end{bmatrix}^T$. A esto se le conoce como una solución particular, aun que no es la única.

Para obtener el resto de soluciones, habrá que ser creativos generando 0s, utilizando las columnas de la matriz. Podemos expresar la tercera columna como una suma de las dos primeras:$$\begin{bmatrix} 8 \\ 2\end{bmatrix} = 
8*\begin{bmatrix} 1 \\ 0\end{bmatrix} + 2*\begin{bmatrix} 0 \\ 1\end{bmatrix}$$
De manera que el vector $O = 8c_1 + 2c_2 - 1c_3 + 0c_4$ y $(x_1,x_2,x_3,x_4) = (8,2,-1,0)$. Cualquier escalar $\lambda \in \Re$  produce el vector $O$. $$\begin{bmatrix}1 & 0 & 8 & -4 \\ 0 & 1 & 2 & 12 \end{bmatrix} * (\lambda \begin{bmatrix} 8 \\ 2\\ -1\\ 0 \end{bmatrix}) = \lambda(8c_1+2c_2-c_3) = O$$
De la misma manera que :$$\begin{bmatrix} -4 \\ 12 \end{bmatrix} = 
-4*\begin{bmatrix} 1 \\ 0\end{bmatrix} + 12*\begin{bmatrix} 0 \\ 1\end{bmatrix} $$$$\begin{bmatrix}1 & 0 & 8 & -4 \\ 0 & 1 & 2 & 12 \end{bmatrix} * (\lambda \begin{bmatrix} -4 \\ 12\\ 0\\ -1 \end{bmatrix} ) = \lambda(-4c_1+12c_2-c_4) = O$$
Y combinándolo todo, obtenemos:$$x  ºin \Re^4 : x = \begin{bmatrix} 42 \\ 8\\ 0\\ 0 \end{bmatrix} + (\lambda_1*\begin{bmatrix} 8 \\ 2\\ -1\\ 0 \end{bmatrix}) + (\lambda_2 \begin{bmatrix} -4 \\ 12\\ 0\\ -1 \end{bmatrix} )$$
Resumiendo, los pasos a seguir para resolver sistemas de ecuaciones mediante matrices son los siguientes.
1. Encontrar una solución particular para $Ax = b$
2. Encontrar las soluciones para $Ax = 0$
3. Combinar las soluciones para obtener una general

Aun que esta matriz tenía una particularidad en las dos primeras columnas, para poder resolver matrices mas complejas, primero tendremos que simplificarlas con el método Gaussiano de eliminación.

### Transformaciones elementales
Para simplificar las matrices podemos hacer lo siguiente:
- Intercambiar ecuaciones
- Multiplicar una ecuación por una $\lambda \in \Re  \neq {0}$  
- Sumar ecuaciones

Siendo $a \in \Re$, buscamos las soluciones al sistema siguiente:$$\begin{cases}
  -2x_1 + 4x_2 - 2x_3  -x_4 + 4x_5 = -3 \\
  4x_1 - 8x_2 + 3x_3  - 3x_4 + x_5 = 2 \\
  x_1 - 2x_2 + x_3  - x_4 + 4x_5 = 0 \\
  x_1 - 2x_2        - 3x_4 + 4x_5 = a
\end{cases}$$

$$\begin{bmatrix}-2 && 4 && -2 && 1 && -4 && |-3 \\
                 4 && -8 && 3 && -3 && 1 && | 2\\
                1 && -2 && 1 && -1 && 1 && | 0\\
                1 && -2 && 0 && 4 && 4 && | a\\\end{bmatrix}$$

Habiendo intercambiado $R_1 \leftrightarrow R_3$ y con las siguientes simplificaciones:
- $R_2 - 4R_1;*-(1)$
- $R_3 + 2R_1;*(-\frac{1}{3})$
- $R_4 -R_1:R_4-R_2-R_3$

### Fila-escalon
Obtenemos la siguiente matriz:
$$\begin{bmatrix}1 && -2 && 1 && -1 && 1 && |0 \\
                 0 && 0 && 1 && -1 && 3 && | -2\\
                0 && 0 && 0 && 1 && -2 && | 1\\
                0 && 0 && 0 && 0 && 0 && | a+1\\\end{bmatrix}$$

Con $a=-1$ como solución parcial:$$\begin{bmatrix}x_1\\ x_2\\ x_3\\ x_4\\ x_5\\ \end{bmatrix} = \begin{bmatrix}2\\ 0\\ -1\\ 1\\ 0\\ \end{bmatrix}$$
Y la solución general: $$x \in \Re^5 : x = \begin{bmatrix}2\\ 0\\ -1\\ 1\\ 0\\ \end{bmatrix} + \lambda_1\begin{bmatrix}2\\ 1\\ 0\\ 0\\ 0\\ \end{bmatrix} + \lambda_2\begin{bmatrix}2\\ 0\\ -1\\ 2\\ 1\\ \end{bmatrix}$$
#### Importante
Fijas los pivotes como variables inmutables y buscas las soluciones en base a las variables libres. 
$x_1, x_3, x_4 \rightarrow \text{Pivotes}$
$x_2,x_5 \rightarrow \text{Variables libres}$

### Calculando la inversa
Para calcular la matriz $A^{-1}$ de $A \in \Re^{n*n}$ necesitamos encontrar una matriz $X$ tal que $AX=I_n$, entonces $X=A^{-1}$. Podemos expresarla de tal forma que $AX=I_n$, Siendo $X=[x_1],....,[x_n]$ ecuaciones lineales que solucionamos.

Por lo que, si expresamos la matriz en formas de filas-escalón, podemos leer la inversa a la derecha de la ecuación, asi como $[A|I_n] \rightarrow ... \rightarrow [I_n|A]$ $$\begin{bmatrix} 
1 && 0 && 2 && 0| 1 && 0 && 0 && 0
\\ 1 && 1 && 0 && 0| 0 && 1 && 0 && 0
\\ 1 && 2 && 0 && 1| 0 && 0 && 1 && 0
\\ 1 && 1 && 1 && 1| 0 && 0 && 0 && 1
\end{bmatrix} \rightarrow \begin{bmatrix} 
1 && 0 && 0 && 0| -1 && 3 && -2 && 2
\\ 0 && 1 && 0 && 0| 1 && -1 && 2 && -2
\\ 0 && 0 && 1 && 0| 1 && -1 && 1 && -1
\\ 0 && 0 && 0 && 1| -1 && 0 && 1 && 2
\end{bmatrix}$$
### Algoritmos para resolver Sistemas de ecuaciones lineales
Si no se encuentra una solución que satisfaga la ecuación tal que $A_x = b$, debemos aproximar las soluciones:
- Regresión lineal
- $A^{-1}$ tal que $A_x = b \rightarrow x= A^{-1}b \in \Re^{n*n}$
- $A_x = b \leftrightarrow A^TA_x = A^Tb \leftrightarrow x(A^TA)^{-1}A^Tb$

