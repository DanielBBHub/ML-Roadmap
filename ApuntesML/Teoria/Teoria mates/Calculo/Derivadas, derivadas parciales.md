---
tipo: concepto
categoria: matematicas
subcategoria: calculo
tags:
  - derivadas
  - derivadas-parciales
  - diferenciacion
  - tasa-cambio
seccion_roadmap: Mathematical Foundations
orden_seccion: 1
orden: 6
dificultad: media
---
## Derivadas
En esta sección se describirá como calcular los gradientes de las funciones, esenciales en ML, ya que estos apuntan al mayor ascenso en la función

### Diferenciación de funciones univariantes
Empezamos con el cociente diferencial de una función univariante $y = f(x), x, u \in \Re$
#### Definición 1: Cociente diferencial
$$\frac{\partial y}{\partial x} := \frac{f(x + \partial x) - f(x)}{\partial x}$$
Esta ecuación calcula la pendiente de la línea secante definida por dos puntos como coordenadas $\large x, x_0,x_0 + \partial x$.

El cociente diferencial también se puede considerar la media de la pendiente entre $[x,x + \partial x]$ asumiendo que la función es lineal

#### Definición 2: Derivada
Formalmente, para $h>0$, la derivada de $f$ en $x$ es definida como el límite:$$\frac{df}{dx}:= \lim_{h \rightarrow 0} \frac{f(x + h) - f(x)}{g}$$
##### Ejemplo: Derivada de un polinomio
Queremos calcular la derivada del polinomio $f(x) = x^n ,n\in \mathbb{N}$:
$\large \frac{df}{dx} = \lim_{h \rightarrow 0} \frac{(x+h)^n - x^n}{h}$
$$ \begin{aligned}
\large \frac{df}{dx} = \lim_{h \rightarrow 0} \frac{(x+h)^n - x^n}{h} = \\
\large = \lim_{h \rightarrow 0} \binom{n}{1}x^{n-1} + \sum^n_{i=2} \binom{n}{i}x^{n-i}+h^{i-1} =\\
= \lim_{h \rightarrow 0} \binom{n}{1}x^{n-1}; \frac{df}{dx} = nx^{n-1}
\end{aligned} $$

### Las series de Taylor
La serie de Taylor es una representación de una función $f$ como una suma infinita de elementos, determinados por derivadas de $f$ en $x_0$

### Definición 3: Polinomio de Taylor
El polinomio de Taylor de grado $n$ de $f:\Re \rightarrow \Re$ en $x_0$ se define como:$$T_N(x) = \sum^n_{k=0}\frac{f^{(k)(x_0)}}{k!}(x-x_0)^k$$
donde $f^{(k)}$ es la k-derivada de $f$ en $x_0$ y $\large \frac{f^{(k)(x_0)}}{k!}$ son los coeficientes del polinomio

### Definición 4: Series de Taylor:
Para una función siave $f \in C^∞$ $f: \Re \rightarrow \Re$, la serie de Taylor para $f$ en $x_0$ es: $$T_∞(x) = \sum^∞_{k=0}\frac{f^{(k)(x_0)}}{k!}(x-x_0)^k$$
#### Ejemplo: Polinomio de Taylor:
Dada $f(x) = x^4$, queremos calcular $T^6$ para $x_0=1$, es decir, $f^{(k)}(1)$ para $k=0,...,6$: $$\begin{align}
f(1) &= 1  \\
f'(1) &= 4  \\
f''(1) &= 12  \\
f^{(3)}(1) &= 24  \\
f^{(4)}(1) &= 24  \\
f^{(5)}(1) &= 0  \\
f^{(6)}(1) &= 0 
\end{align}$$

Por lo tanto, el polinomio de Taylor $$\begin{align}
T_6(x) &= \sum_{k=0}^6 \frac{f^{(k)}(x_0)}{k!} (x - x_0)^k \tag{5.17a} \\
&= 1 + 4(x - 1) + 6(x - 1)^2 + 4(x - 1)^3 + (x - 1)^4 + 0 \tag{5.17b}
\end{align}$$

### Reglas de la diferenciación
- Producto:$$\large (f(x)g(x))' = f'(x)g(x)+f(x)g'(x)$$
- Cociente: $$\large (\frac{f(x)}{g(x)})' = \frac{f'(x)g(x)-f(x)g'(x)}{(g(x))^2}$$
- Suma: $$\large(f(x)+g(x))' = f'(x) + g'(x) $$
- Cadena: $$\begin{aligned}\large ((g(f(x)))' = (g◦f)'(x) = g'(f(x))f'(x) \\ x\rightarrow f(x)\rightarrow g(f(x))\end{aligned}$$
#### Ejemplo: Regla de la cadena
Calculemos la derivada de la función \( h(x) = (2x + 1)^4 \) utilizando la regla de la cadena. Con
$$\begin{align}
h(x) &= (2x + 1)^4 = g(f(x)), \\
f(x) &= 2x + 1, \\
g(f) &= f^4,
\end{align}$$
Siguiendo la regla de la cadena $\large (g(f(x)))' = g'(f(x))f'(x)$ $$\begin{align}
f'(x) &= 2, \\
g'(f) &= 4f^3,
\end{align}$$
$$\begin{align}
h'(x) &= g'(f) \, f'(x)  \\
&= (4f^3) \cdot 2  \\
&= 4(2x + 1)^3 \cdot 2  \\
&= 8(2x + 1)^3 
\end{align}$$

## Derivadas parciales
Hasta ahora la diferenciación solo ha tenido en cuenta funciones que dependian de una sola variable $x \in \Re$. La generalización de la derivada de las funciones multivariables son los gradientes .

Se calcula el gradiente de una función $T$ con respecto a $x$ variando una variable mientras se dejan constantes el resto, lo cual hace que el gradiente sea el conjunto de derivadas parciales

### Definición 5: Derivadas parciales:
Para una función $f: \Re^n \rightarrow \Re, x \rightarrow f(x) x\in \Re^n$, con n variables, definimos las derivadas parciales como:$$\begin{align}
\frac{\partial f}{\partial x_1} &= \lim_{h \to 0} \frac{f(x_1 + h, x_2, \ldots, x_n) - f(\mathbf{x})}{h} \notag \\
&\vdots \notag \\
\frac{\partial f}{\partial x_n} &= \lim_{h \to 0} \frac{f(x_1, \ldots, x_{n-1}, x_n + h) - f(\mathbf{x})}{h} \tag{5.39}
\end{align}$$
Y juntarlas todas en el vector-fila:$$\begin{align}
\nabla_{\mathbf{x}} f = \operatorname{grad} f = \begin{bmatrix}
\dfrac{\partial f(\mathbf{x})}{\partial x_1} &
\dfrac{\partial f(\mathbf{x})}{\partial x_2} &
\cdots &
\dfrac{\partial f(\mathbf{x})}{\partial x_n}
\end{bmatrix} \in \mathbb{R}^{1 \times n}, \tag{5.40}
\end{align}$$
Donde $n$ es el número de variables y $1$ es la dimensión de la imágen/rango/codominio de $f$.

#### Ejemplo: Derivadas parciales con la regla de la cadena
Dada $f(x,y) = (x+2y^3)^2$, obtenemos las siguientes derivadas parciales:$$\begin{align}
\frac{\partial f(x, y)}{\partial x} &= 2(x + 2y^3)\frac{\partial}{\partial x}(x + 2y^3) = 2(x + 2y^3), \tag{5.41} \\
\frac{\partial f(x, y)}{\partial y} &= 2(x + 2y^3)\frac{\partial}{\partial y}(x + 2y^3) = 12(x + 2y^3)y^2, \tag{5.42}
\end{align}$$

#### Ejemplo: Gradiente
Dada $f(x_1,x_2) = x_1^2 x_2 + x_1x_2^3 \in \Re$, las derivadas parciales son: $$\begin{align}
\frac{\partial f(x_1, x_2)}{\partial x_1} &= 2x_1x_2 + x_2^3 \tag{5.43} \\
\frac{\partial f(x_1, x_2)}{\partial x_2} &= x_1^2 + 3x_1x_2^2 \tag{5.44}
\end{align}$$resultando en un gradiente $$\begin{align}
\frac{d\mathbf{f}}{d\mathbf{x}} = \left[ \frac{\partial f(x_1, x_2)}{\partial x_1}, \frac{\partial f(x_1, x_2)}{\partial x_2} \right] = \bigl[ 2x_1x_2 + x_2^3,\; x_1^2 + 3x_1x_2^2 \bigr] \in \mathbb{R}^{1 \times 2}. \tag{5.45}
\end{align}$$

### Reglas básicas de las derivadas parciales
- Producto: $$
\frac{\partial}{\partial x} (f(x)g(x)) = \frac{\partial f}{\partial x}g(x) + f(x)\frac{\partial g}{\partial x} 
$$
- Suma:$$
\frac{\partial}{\partial x}(f(x) + g(x)) = \frac{\partial f}{\partial x} + \frac{\partial g}{\partial x} 
$$

- Resta: 
$$\frac{\partial}{\partial x}(f(x) - g(x)) = \frac{\partial f}{\partial x} - \frac{\partial g}{\partial x}$$

- Cadena
$$\frac{\partial}{\partial x}(g \circ f)(x) = \frac{\partial}{\partial x}(g(f(x))) = \frac{\partial g}{\partial f} \frac{\partial f}{\partial x} $$

- Cociente: 
$$\frac{\partial}{\partial x}\left(\frac{f(x)}{g(x)}\right) = \frac{\frac{\partial f}{\partial x}g(x) - f(x)\frac{\partial g}{\partial x}}{(g(x))^2}, \quad g(x) \neq 0$$

### Regla de la cadena
Dada una función $f:\Re^2 \rightarrow \Re$ con variables $[x_1,x_2]$. Además, $x_1(t) y x_2(t)$ son funciones de $t$. Para calcular el gradiente de $f$ con respecto a $t$ aplicamos la regla de la cadena para funciones multivariables como: $$\begin{align}
\frac{df}{dt} &= \left[ \frac{\partial f}{\partial x_1} \quad \frac{\partial f}{\partial x_2} \right]
\begin{bmatrix}
\frac{\partial x_1(t)}{\partial t} \\[4pt]
\frac{\partial x_2(t)}{\partial t}
\end{bmatrix}
= \frac{\partial f}{\partial x_1} \frac{\partial x_1}{\partial t} + \frac{\partial f}{\partial x_2} \frac{\partial x_2}{\partial t}, \tag{5.49}
\end{align}$$
donde $d$ representa gradiente y $\partial$ derivadas parciales

#### Ejemplo:
Dada la funcion $( f(x_1, x_2) = x_1^2 + 2x_2 ),$ donde $( x_1 = \sin t )$ y $( x_2 = \cos t)$, entonces
$$\begin{align}
\frac{df}{dt} &= \frac{\partial f}{\partial x_1} \frac{\partial x_1}{\partial t} + \frac{\partial f}{\partial x_2} \frac{\partial x_2}{\partial t}  \\
&= 2 \sin t \frac{\partial \sin t}{\partial t} + 2 \frac{\partial \cos t}{\partial t}  \\
&= 2 \sin t \cos t - 2 \sin t = 2 \sin t (\cos t - 1) 
\end{align}$$

Si $f(x_1,x_2)$ es una función de $x_1$ y $x_2$, donde $x_1(s,t)$ y $x_2(s,t)$ son ellas mismas también funciones, la regla de la cadena nos devuelve las siguientes derivadas parciales: $$\begin{align}
\frac{\partial f}{\partial s} &= \frac{\partial f}{\partial x_1} \frac{\partial x_1}{\partial s} + \frac{\partial f}{\partial x_2} \frac{\partial x_2}{\partial s},  \\
\frac{\partial f}{\partial t} &= \frac{\partial f}{\partial x_1} \frac{\partial x_1}{\partial t} + \frac{\partial f}{\partial x_2} \frac{\partial x_2}{\partial t}, 
\end{align}$$

y el gradiente se obtiene de la siguiente forma: $$\begin{align}
\frac{d\mathbf{f}}{d(s, t)} &= \frac{\partial \mathbf{f}}{\partial \mathbf{x}} \frac{\partial \mathbf{x}}{\partial (s, t)} = \begin{bmatrix} \dfrac{\partial \mathbf{f}}{\partial x_1} & \dfrac{\partial \mathbf{f}}{\partial x_2} \end{bmatrix} \begin{bmatrix} \dfrac{\partial x_1}{\partial s} \\[6pt] \dfrac{\partial x_2}{\partial s} \end{bmatrix}, \tag{5.53}
\end{align}$$
$$\begin{align}
&= \frac{\partial \mathbf{f}}{\partial \mathbf{x}} \quad \text{and} \quad \frac{\partial \mathbf{x}}{\partial (s, t)} = \frac{\partial \mathbf{f}}{\partial (s, t)}.
\end{align}$$