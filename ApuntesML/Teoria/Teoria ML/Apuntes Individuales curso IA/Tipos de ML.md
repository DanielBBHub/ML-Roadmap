## Tipos de sistemas de ML

Hay varios, con lo que es útil categorizarlos siguiendo los siguientes criterios:
- Si están o no entrenados con supervisión humana (aprendizaje supervisado, no supervisado, semisupervisado y por refuerzo)
- Si pueden aprender de manera incremental "al vuelo" (aprendizaje por lotes, en línea)
- Si trabajan comparando nueva información o si detectan patrones en la información de entrenamiento y se construye un modelo predictivo (aprendizaje basado en instancias, basado en modelos)

### Aprendizaje supervisado
En el aprendizaje supervisado, la información de entrenamiento que se le da al algorítmo incluye la solución deseada, conocidas como etiquetas.

Un uso típico de esta aplicación son los filtros de spam: se entrena con muchos correos de ejemplo, etiquetados con su clase, con el objetivo de que sea capaz de diferenciar los correos normales a aquellos que son spam

Los algorítmos de aprendizaje supervisado mas famosos son:
- K-NN
- Regresión lineal
- Regresión logística
- SVMs
- Árboles de decisión / Random Forest
- Algunas redes neuronales

### Aprendizaje no supervisado
En este caso, la información de entrenamiento no está etiquetada con la clase correspodiente, el sistema "intenta aprender sin un profesor".

Los algorítmos de aprendizaje no supervisado mas famosos son:
- Clustering
  - K-Means
  - Análisis jerárquico de clusters (HCA)
  - Maximización de la expectativa
- Visualización y reducción de dimensiones
  - Análisis del componente principal (PCA)
  - Kernel PCA
  - Locally.linear embedding (LLE)
  - t-distributed Stochastic Neighbor Embedding (t-SNE)
- Aprendizaje por asociación de reglas
  - Apriori
  - Eclat

### Aprendizaje semi-supervisado
Algunos algorítmos puedes lidiar con información de entrenamiento que esta parcialmente etiquetada, normalmente la mayor parte de la información no contiene etiqueta.

Este es el caso de Google Photos, el cual es capaz de reconocer la misma persona A en las fotos 1, 4, 20 y a la persona B en fotos 4, 7, 18 de un conjunto de imágenes. Esta parte es aprendizaje no supervisado y lo que resta es que se le diga al sistema quienes son A y B para que sea capaz de nombrarlos en cualquiera de las imágenes en las que aparezcan.

### Aprendizaje por refuerzo
Este sistema es bastante diferente a los anteriores. El sistema de aprendizaje, llamado agente en este contexto, puede observar el entorno, seleccionar y realizar acciones y recibir recompensas positivas/negativas dependiendo de la estrategia. Por lo tanto, el sistema ha de llegar a la mejor estratégia (política) para alcanzar la mayor cantidad de recompensas.

El flujo de aprendizaje luce de la siguiente manera:
1. Observar
2. Seleccionar política
3. Realizar acción
4. Recibir recompensa/penalización
5. Actualizar política (paso de aprendizaje)
6. Iterar hasta llegar a una política óptima

### Aprendizaje por lotes 
El sistema es incapaz de aprender incrementalmente; se ha de entrenar con toda la información disponible. Normalmente, este hecho conlleva tiempo y capacidad de computación elevadas, con lo que se suele hacer en local. Primero se entrena el sistema y luego se lanza a producción, sin aprendizaje, aplica lo aprendido previamente.

Para añadir mas información a la base de conocimiento del modelo, se ha de volver a entrenar con el dataset completo; la información con la que ya se había entrenado mas la información nueva.

### Aprendizaje en línea / incremental
Con este método se entrena al sistema dándole instancias de información secuencialmente, ya se aindividualmente o en conjuntos pequeños. Cada paso del aprendizaje es rápido y barato, con lo que puede actualizar la información al vuelo tan rápido como se genera

### Aprendizaje basado en instancias



### Aprendizaje basado en modelos