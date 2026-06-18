## Diseño SCIKIT-LEARN

La API de la librería apunta a tener un diseño con las siguientes máximas:
- Consistencia
    - "Estimators": Cualquier objeto que pueda estimar parametros basados en un dataset (p.j: SimpleImputer). La estimación en si se realiza con "fit()", únicamente recibiendo un dataset como entrada.
    - "Transformers": Algunos estimadores pueden también transformar un dataset, estos son los "transformers". La transformación se realiza con "transform()" con el dataset a transformar como parámetro. Es importante apuntar que la transformación se realiza en base a los parametros aprendidos. Finalmente, estas clases tienen una función "fit_transform()" que es equivalente a llamar a "fit()" y despues a "transform()" aun que algunas clases tienen la función general optimizada
    - "Predictors": Algunos estimadores tienen la capacidad de realizar predicciones sobre un dataset, los "predictors". Estas clases tienen un método "predict()" que recibe un dataset de nuevas instancias y devuelve una predicción. Por otro lado tiene un método "score()" que mide la calidad de la predicción sobre un dataset.

- Inspección/Transparencia: Todos los hyperparametros de los estimadores son accesibles directamente mediante las variables de la instancia (p.j: imputer_strategy), de la misma manera que todos los parametros aprendidos (p.j. imputer.statistics_)

- No proliferación de clases: Los datasets se representan como arrays de NumPy o "sparse matrices" de SciPy

- Composición: Máxima reutilización de los cimientos de la libreria

- "Por-defectos" razonables: La librería provee valores por defecto razonables para la mayoría de parametros