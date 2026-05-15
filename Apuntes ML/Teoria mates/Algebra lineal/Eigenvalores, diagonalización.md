---
tipo: concepto
categoria: matematicas
subcategoria: algebra-lineal
tags:
  - eigenvalores
  - eigenvectores
  - diagonalizacion
  - autovalores
seccion_roadmap: Mathematical Foundations
orden_seccion: 1
orden: 4
dificultad: media
---
## Eigenvalores y eigenvectores
Es una forma diferente de caracterizar matrices y su asociación lineal. Los "eigenvalores" nos explicarán como se transforman los "eigenvectores".

Dada la matriz $A \in \Re^{n*n}$, entonces $\lambda \in \Re$ es un eigenvalor de $A$ y $x \in \Re^n \{0\}$ es su eigenvector $$Ax = \lambda x$$
Por convención los eigenvalores se ordenan de manera descendiente. 
Las siguientes premisas son equivalentes:
- $\lambda$ es eigenvalor de $A \in \Re^{n*n}$
- Exist una $x \in \Re^n \{0\}$ con $Ax = \lambda x$
- $rk(A - \lambda I_n) < n$
- $det ( A - \lambda I_n) = 0$

### Definición 1: Colinealidad y codirección
Dos vectores que apuntan a la misma dirección son llamados codirigidos, mientras que dos vectores son colineales si apuntan en la misma dirección o en la opuesta

### Teorema 1
$\lambda \in \Re$ es un eigenvalor de una matriz cuadrada si y solo si $\lambda$ es una raíz del polinomio característico $p_a(\lambda)$ de $A$.

### Definición 2
Dada $A \in \Re^{n*n}$ y su eigenvalor $\lambda$, la multiplicación algebraica de $\lambda_i$ es el número de veces que esta raíz aparece en $p_a(d)$ de $A$

### Definición 3: Eigenespacio y eigenespectro
Dada $A \in \Re^{n*n}$, el conjunto de todos los eigenvectores asociados con $A$ con un eigenvalor $\lambda$  ocupan un subespacio de $R^n$ llamado eigenespacio de $A$ con respecto de $\lambda$ y se expresa como $E_\lambda$. El conjunto de todos los eigenespacios se denomina eigenespectro.

Si $\lambda$ es un eigenvalor de $A \in \Re^{n*n}$, entonces su egienespacio es el espacio de soluciones del sistema de ecuaciones lineales $(A - \lambda I_n)=0$

### Propiedades
- La matriz $A$ y $A^T$ tienen los mismos eigenvalores pero puede que no los mismos eigenvectores
- El eigenespacio $E_\lambda$ es el espacio nulo de $A - \lambda I$: $$\large Ax \leftrightarrow Ax - \lambda x = 0 \leftrightarrow (A - \lambda I)=0 \leftrightarrow x \in ker(A \lambda I)$$
- Las matrices finitas positivas siempre tienen eigenvalores positivos

Ejemplo: Dada la matriz $A = \begin{bmatrix} 4 && 2 \\ 1 && 3 \end{bmatrix}$:
- Paso 1: $p_A(\lambda)$
	- $p_A(\lambda) = det (A- \lambda i)$
	- $p_A(\lambda) = \begin{bmatrix} 4 && 2 \\ 1 && 3 \end{bmatrix} - \begin{bmatrix} \lambda && 0 \\ 0 && \lambda \end{bmatrix} = \begin{bmatrix} 4 - \lambda && 2 \\ 1 && 3 -\lambda \end{bmatrix}$
	- $P_A(\lambda )= (4 - \lambda)(3- \lambda) - 2 = ( 12 - 4\lambda - 3\lambda + \lambda^2)$
- Paso 2: raices $\lambda_1 = 2 \lambda_2 = 5$
- Paso 3: eigenvectores y eigenespacios
	- Encontramos los eigenvectores de los eigenvalores anteriores por el vector $x$ $$\begin{bmatrix} 4 - \lambda && 2 \\ 1 && 3 -\lambda \end{bmatrix} x = 0$$Para $\lambda = 5$ $$\begin{bmatrix} 4 - 5 && 2 \\ 1 && 3 - 5 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} -1 && 2 \\ 1 && -2 \end{bmatrix} = 0$$La solución es $span(\begin{bmatrix} 2 \\ 1 \end{bmatrix})$ 
	  Para $\lambda = 2$ $$\begin{bmatrix} 4 - 2 && 2 \\ 1 && 3 - 2 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} 2 && 2 \\ 1 && 1 \end{bmatrix} = 0$$La solución es $span(\begin{bmatrix} 1 \\ -1 \end{bmatrix})$ 
Los eigenespacioes son vectores unidimensionales

En este ejemplo $E_2$ y $E_5$ son unidimensionales ya que ambos se ven representados con un único ector, aun que en otros casos, estos pueden tener mas de un eigenvalor idéntico y ser multidimensionales

### Definición 5
Dado $\lambda_i$ como eigenvalor de una matriz cuadrada, entonces la multiplicidad geométrica de $\lambda_i$ es el número de eigenvectores linealmente independientes asociados con $\lambda_i$. Es decir, es la dimensión del eigenespacio asociado con $\lambda_i$: 
- Dada $A = \begin{bmatrix} 2 && 1  \\ 0 && 2\end{bmatrix}$, $\lambda_1 = \lambda_2 = 2$. Esta tiene una multiplicidad algebraica de 2, pero el eigenvalor tiene un único eigenvector $ x= \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ con lo que la multiplicidad geométrica es 1

### Intuición gráfica en 2D
![[Pasted image 20260508093012.png]]
- $A_1 = \begin{bmatrix} \frac{1}{2} && 0 \\ 0 && 2 \end{bmatrix}$ describe la dirección de dos eigenvectores en los dos ejes cordinales. El eje vercial se extiende por un eigenvalor $\lambda = 2$ mientras que el horizontal se comprime por un $\lambda \frac{1}{2}$. El área mapeada $det(A_1) = 1 = 2 * \frac{1}{2}$
- $A_2 = \begin{bmatrix} 1 && \frac{1}{2}  \\ 0 && 1 \end{bmatrix}$ traslada los puntos situados en el eje horizontal hacia la derecha si están en la mitad positiva del eje vertical y a la izquierda si están en el otro lado.
- $A_3 = \begin{bmatrix} \cos(\frac{\pi}{6}) && -\sin(\frac{\pi}{6}) \\ \sin(\frac{\pi}{6}) && \cos(\frac{\pi}{6}) \end{bmatrix} = \frac{1}{2} \begin{bmatrix} \sqrt{3} && -1 \\ 1 && \sqrt{3} \end{bmatrix}$ La matriz $A_3$ rota los puntos por $\frac{\pi}{6}rad = 30º$ en dirección contraria a las manecillas del reloj y solo tiene eigenvalores complejos. Esta rotación tiene que preservar el volumen
- $A_4 = \begin{bmatrix} 1 && -1 \\ -1 && 1 \end{bmatrix}$ representa un mapeo que colapsa el dominio $2D \rightarrow 1D, (\lambda_1 \approx \lambda_2)$. por lo que deja a un eigenvector $\lambda_1 = 0$, mientras que $\lambda_2 = 2$ amplia el espacio
- $A_5 = \begin{bmatrix} 1 && \frac{1}{2} \\ \frac{1}{2} && 1 \end{bmatrix}$ es una traslación-estirado de los puntops, que escala el espacio un 75%, ya que $det(a) = \frac{3}{4}$. Se estira a lo largo del eigenvalor de $\lambda_2 = 1.5$ y se comprimer a lo largo de $\lambda_1 = 0.5$

### Teorema 2
Los eigenvectores $x_1, ...,x_n$ de una matriz $A \in \Re^{n*n}$ con $n$ eigenvalores diferentes $\lambda_1, ..., \lambda_n$ son linealmente independientes

### Definición 5
Una matriz $A \in \Re^{n*n}$ es defectuosa si tiene menos de n eigenvalores linealmente independientes

### Teorema 3
Dada $A \in \Re^{n*n}$ , siempre podemos obtener una amtriz simétrica, positiva y semidefinitiva $S \in \Re^{n*n}$  definidiendo: $$S:= A + A^T \text{ si } rk(A)=n$$
### Teorema 4: Teorema espectral
Si $A \in \Re^{n*n}$ es simétrica, existe una base ortonormal para el correspondiente espacio vectorial $V$ que consiste en eigenvetores de $A$, todos con eigenvalores positivos.

### Teorema 5
El determinante de una matriz cuadrada es el producto de sus eigenvalores $$det(A) = \prod_{i=1}^n \lambda_i$$donde $\lambda_i \in ℂ$ son eigenvalores de A

### Teorema 6
La traza de una matriz cuadrada es la suma de sus eignevalores $$tr(A) = \sum_{i=1}^n \lambda-i$$donde $\lambda_i \in ℂ$ son eigenvalores de A
#### Ejemplo: Ranking de páginas de google
Google utiliza el eigenvector correspondiente a la matriz $A$ para determinar el ranking de la página.

La idea es, fundamentalmente, que la importancia de una página puede aproximarse a la cantidad de páginas enlazadas a esta. Se representan las webs en un grafo y se calcula el peso de cada una con los enlaces a esta.

Por otro lado se caracteriza la navegación de un usuario mediante una matriz de transferencia $A$ del grafo que indica con que probabilidad acabará en otra página el usuario.

## Diagonalización

### Descomposición de Cholesky
La descomposición de Cholesky nos permite descomponer en factores más pequeños las matrices definitivas positivas.

#### Teorema 1
Una matriz simétrica, positiva puede ser descompuesta en un producto $A = LL^T$, siendo L una amtriz triangular inferior con elementos diagonales positivos$$ 
\begin{bmatrix}
a_{11} & \cdots & a_{1n} \\
\vdots &   & \vdots \\
a_{n1} & \cdots & a_{nn}
\end{bmatrix} = \begin{bmatrix}
l_{11} & \cdots & 0 \\
\vdots &   & \vdots \\
l_{n1} & \cdots & l_{nn}
\end{bmatrix} * \begin{bmatrix}
l_{11} & \cdots & l_{1n} \\
\vdots &   & \vdots \\
0 & \cdots & l_{nn}
\end{bmatrix}$$
Se conoce a $L$ como factor Cholesky y esta matriz es única.$$A = \begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33}
\end{bmatrix} = \begin{bmatrix}
l_{11} & 0 & 0 \\
l_{21} & l_{22} & 0 \\
l_{31} & l_{32} & l_{33}
\end{bmatrix} * \begin{bmatrix}
l_{11} & l_{12} & l_{13} \\
0 & l_{22} & l_{23} \\
0 & 0 & l_{33}
\end{bmatrix}$$$$A = \begin{bmatrix}
l^2_n & l_{21}l_{11} & l_{31}l_{11} \\
l_{21}l_{11} & l^2_{21}l^2_{22} & l_{31}l_{21} + l_{32}l_{22} \\
l_{31}l_{11} & l_{32}l_{22} & l^2_{31} + l^2_{32} + l^2_{33}
\end{bmatrix}$$
De esta forma podemos calcular los valores de los componentes diagonales $\large l_{ij}$ comparándolos con la matriz original $A$ $$l_{11} = \sqrt{a_{11}},l_{22} = \sqrt{a_{22} - l^2_{21}},l_{33} = \sqrt{a_{33}  (l^2_{31}+l^2_{32ç})}$$
Y de la misma manera las que están debajo de la diagonal$$l_{21}$=\frac{1}{l_{11}}a_{21}, l_{21}=\frac{1}{l_{11}}a_{31}, l_{32}=\frac{1}{l_{22}}(a_{32} - l_{31}l_{21})$$
Esta herramienta nos permite manipular las matrices para los cálculos necesarios en ML, como la matriz de covarianza de una variable Gaussiana o el cálculo de los determinantes $\large det(A)=det(L)det(L^T)$ 

### Eigendescomposición y diagonalización

Una matriz diagonal es aquella que tiene $0$s en la diagonal contraria$$D = 
\begin{bmatrix}
c_1 & \cdots & 0 \\
\vdots & \ddots & \vdots \\
0 & \cdots & c_n
\end{bmatrix}$$

Son útiles para el cómputo rápido de determinantes, potencias e inversas. $det(D) es el producto de sus elementos diagonales, la potencia $D^k$ es el resultado de sus diagonales a la $^k$ y la inversia $D^{-1}$ es la recíproca de sus elementos si $\neq 0$

#### Definición 1: Diagonalizable
Una matriz $A \in \Re^{nn}$ es diagonalizable si es similar a una matriz diagonal, es decir, $P \in \Re^{nn}$ de manera que $d = P^{-1}AP$

Veremos que diagonalizar una matriz $A \in \Re{nn}$ es una manera de expresar los mismos mapeados lineales con un cambio de base, que consiste en los eigenvectores de $A$.

Dada $\large A \in \Re{nn} ,\lambda_1, ..., \lambda_n$ de un conjuntode escalares y $\large p_1, ..., p_n$ de un conjunto de vectores $\large \in \Re^{nn}$. Definimos $P:= [p_1,...,p_n]$ y dado $D\in\Re^{nn}$ una matriz diagonal con elementos $\large \lambda_1,..., \lambda_n$, entonces $AP=PD$ si y solo si $\large \lambda_1,..., \lambda_n$ son eigenvalores de $A$ y las columnas de $P$ son eigenvectores de A 

#### Teorema 1: Eigendescomposición
Una matriz cuadrada se puede factorizar en $$A=PDP^{-1}$$
donde $P\in\Re^{nn}$ y $D$ en una diagonal donde la diagonal son eigenvalores de $A$, si y solo si los eigenvectores de $A$ forman una base $\Re^n$.

Este teorema implica que únicamente se puede diagonalizar las matrices no defectuosas y que las columnas de $P$ son un conjunto de eigenvectores de $A$

#### Teorema 2
Una matriz simétrica siempre va a ser diagonalizable

#### Intuición geométrica de la descomposición
Podemos interpretar una eigendescomposición de la siguiente manera:
> Dada la matriz de transformación lineal $A$ con mapeado a la base estandar $\large e_i$ $P^{-1}$ realiza un cambio de base (base estandar $\rightarrow$ eigebase). Luego $D$ escala esta base por un factor $\large \lambda_i$. Finalmente, $P$ transforma los vectores escalados a la base estandar (eigenbase $\rightarrow$ base estandar)

- Las matrices $D$ son eficientes de exponer ya que se aplica la potencia individualmente a los elementos de la diagonal $$A^K = (PDP^{-1})^K = PD^KP^{-1} $$
- Asumiendo que exista eigendescomposición para la matriz $A$:$$det(A) = det(PDP^{-1}) = det(P)det(D)det(P^{-1}) = det(D) = \prod_i d_{ii}$$ lo cual permite calcular eficientemente el $det(A)$