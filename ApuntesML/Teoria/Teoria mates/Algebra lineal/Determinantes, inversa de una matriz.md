---
tipo: concepto
categoria: matematicas
subcategoria: algebra-lineal
tags:
  - determinantes
  - matriz-inversa
  - matriz-regular
  - singular
seccion_roadmap: Mathematical Foundations
orden_seccion: 1
orden: 3
dificultad: media
---
## Inversa de una matriz
![[Matrices y operaciones con matrices#Inversa y traspuesta]]
![[Matrices y operaciones con matrices#Calculando la inversa]]

## Determinantes y trazas
El determinante es un objeto matemático usado en el análisis y solución de sistemas de ecuaciones lineales. Estos son utilizados en matrices cuadradas y se expresan como $det(A) \text{ || } |A|$.

El determinante de una matriz cuadrada $A \in \Re^{n*n}$ es una función que mapea la matriz A a un número real. Ej Para una matriz $1*1$, un escalar, $A = a \rightarrow A^{-1} = \frac{1}{a}; a*\frac{1}{a} = A \text{ si y solo si } a\neq 0$ .

Para matrices $2*2$, sabemos que $AA^{-1} = I$$$A^{-1} = \frac{1}{a_{11}a_{22}-a_{12}a_{21}  } *\begin{bmatrix} a_{11} && a_{12} \\ a_{21} && a_{22}\end{bmatrix} \text{ si y solo si } a\neq 0 $$
Esta cantidad es el determinante de $A \in \Re^{2*2}$ $$det(A) = \begin{vmatrix}  
a_{11} && a_{12} \\ a_{21} && a_{22}  
\end{vmatrix} = a_{11}a_{22}-a_{12}a_{21}$$

### Teorema 1
Para cualquier matriz $A \in \Re^{n*n}$, $A$ es invertible si $det(A) \neq 0$. Exsisten expesiones explicitas de los determinante spara matrices pequeñas:
- n=1$$det(A) = det(a_{11}) = a_{11}$$
- n=2$$det(A) = \begin{vmatrix}  
a_{11} && a_{12} \\ a_{21} && a_{22}  
\end{vmatrix} = a_{11}a_{22}-a_{12}a_{21}$$
- n=3 $$det(A) = \begin{vmatrix}  
a_{11} && a_{12} && a_{13} \\ 
a_{21} && a_{22} && a_{23} \\
a_{31} && a_{32} && a_{33} \\
\end{vmatrix} = a_{11}a_{22}a_{33} + a_{21}a_{32}a_{13} + a_{31}a_{12}a_{23} 
- a_{31}a_{22}a_{13} -a_{11}a_{32}a_{23} -a_{21}a_{12}a_{33}$$
Con lo que para una matriz cuadrada $T$: $$det(T) = \prod_{i=1}^{2} T_{ii} \text{ (producto de sus elementos diagonales)}$$

#### Matriz triangular superior
Es aquella que dada $T \in \Re^{n*n}; T_{ij} = 0$ para $i>j=0$ es decir, es cero debajo de la diagonal de la matriz.

#### Matriz triangular superior
Es aquella que dada $T \in \Re^{n*n}; T_{ij} = 0$ para $i<j=0$ es decir, es cero encima de la diagonal de la matriz.

### Teorema 2: Expansión de Laplace
Para calcular determinantes para $N>3$ podemos calcular los determinantes de una matriz $n*n$ tal que $(n-1)(n-1)$, pudiendo calcular los determinantes esencialmente como submatrices $2*2$. Ej: Dada una matriz $A \in \Re^{n*n}$, entonces para todo $j=1,...,n$:
- **Expansión por columna $j$**:$$det(A) = \sum_{k=1}^n (-1)^{k*j}a_{kj}det(A_k,j)$$
- **Expansión por fila $j$**:$$det(A) = \sum_{k=1}^n (-1)^{k*j}a_{kj}det(A_j,k)$$
Aqui $A_{k,j} \in \Re^{(n-1)(n-1)}$ es la submatriz de $A$ cuando eliminamos las filas y columnas $k$ y $j$. Ej: Dada la matriz $A \in \Re^{3*3}$$$A = 
\begin{bmatrix} 
1 && 2 && 3 \\ 
3 && 1 && 2 \\ 
0 && 0 && 1 \end{bmatrix} $$
Usando la expansión de laplace por columna: $$\begin{vmatrix} 
1 && 2 && 3 \\ 
3 && 1 && 2 \\ 
0 && 0 && 1 \end{vmatrix} = 
(-1)^{1+1}*1\begin{vmatrix} 1 && 2 \\ 0 && 1\end{vmatrix} + (-1)^{1+2}*2\begin{vmatrix} 3 && 2 \\ 0 && 1\end{vmatrix} + (-1)^{1+3}*3\begin{vmatrix} 3 && 1 \\ 0 && 1\end{vmatrix}$$

### Propiedades
- $det(AB) = det(A)det(B)$
- $det8A) = det(A^T)$
- Si $A$ es invertible $\rightarrow det(A^{-1}) = \frac{1}{det(A)}$
- Añadir columnas/filas que sean múltiples no cambia $det(A)$
- Multiplicar una columna/fila por $\lambda \in \Re$ escala el determinante $det(A) por $\lambda$ 
- Intercambiar filas/columnas cambia el signo de $det(A)$

### Teorema 3
Una matriz cuadrada $A \in \Re^{n*n}$ tiene $det(A) \neq 0$ si y solo si $rk(A) = n$. Es decir, es inertible si y solo si tiene el rango completo.
>$rk(A)$ es el número de filas/columnas linealmente independientes (los vectores son diferentes)

$$\begin{bmatrix} 1&& 3 \\ 0 && 1 \end{bmatrix} \text{ Linealmente independiente, } rk(A) = 2$$
$$\begin{bmatrix} 1&& 2 \\ 2 && 4 \end{bmatrix} \text{ Linealmente dependiente, }rk(A) = 1$$

### Teorema 4
La traza de una amtriz cuadrada $\Re^{n*n}$ se define como $$tr(A):=\sum_{i=1}^n a_{ii}$$
#### Propiedades 
- $tr(A+B) = tr(A) + tr(B)$ para $A,B \in \Re^{n*n}$
- $tr(\alpha *A) = \alpha* tr(A), \alpha \in \Re , A \in \Re^{n*n}$
- $tr(I_n) = n$
- $tr(AB) = tr(BA); A,B \in \Re^{n*n}$
De la misma manera, la traza del producto de una función es invariable a permutaciones cíclicas$$tr(AKL) = tr(KLA); A\in\Re^{a*k},K\in\Re^{k*l},L\in\Re^{l*a}$$

### Definición: Polinomio característico
Dadas $\lambda \in \Re$ y $A \in \Re^{n*n}$:$$p_A(\lambda):= det(A-\lambda I)= c_0 +c_1\lambda + ... + c_{n-1} \lambda^{n-1} + (-1)^n\lambda ^n$$
siendo $c_o,...,c_{n-1}$ el polinomio característico de A
- $c_0 = det(A)$
- $c_{n-1} = (-1)^{n-1} tr(A)$

Este polinomio característico nos permitirá calcular los "Eigenvalores y "Eigenvectores" de una matriz.

