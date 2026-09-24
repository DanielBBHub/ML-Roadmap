# Ejercicios

### Ejercicio 1

¿Cuáles son las ventajas y los inconvenientes de usar una RNN con estado (*stateful*) frente a una RNN sin estado (*stateless*)?

### Ejercicio 2

¿Por qué se usan RNNs encoder-decoder en lugar de RNNs secuencia-a-secuencia simples para la traducción automática?

### Ejercicio 3

¿Cómo se pueden manejar secuencias de entrada de longitud variable? ¿Y secuencias de salida de longitud variable?

### Ejercicio 4

¿Qué es la búsqueda en haz (*beam search*) y por qué la usarías? ¿Qué herramienta puedes usar para implementarla?

### Ejercicio 5

¿Qué es un mecanismo de atención? ¿En qué ayuda?

### Ejercicio 6

¿Cuándo necesitarías usar *sampled softmax*?

### Ejercicio 7

Hochreiter y Schmidhuber usaron las gramáticas de Reber embebidas (*embedded Reber grammars*) en su artículo sobre las LSTM. Son gramáticas artificiales que producen cadenas como "BPBTSXXVPSEPE". Consulta la buena introducción de Jenny Orr a este tema, elige una gramática de Reber embebida concreta (como la que aparece representada en la página de Orr) y entrena una RNN para identificar si una cadena respeta esa gramática o no. Primero tendrás que escribir una función capaz de generar un lote de entrenamiento que contenga aproximadamente un 50 % de cadenas que respeten la gramática y un 50 % que no.

### Ejercicio 8

Entrena un modelo encoder-decoder capaz de convertir una fecha escrita en un formato a otro formato (por ejemplo, de "April 22, 2019" a "2019-04-22").
