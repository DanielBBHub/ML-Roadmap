# Import del dataset de info de inmuebles de california
from sklearn.datasets import fetch_california_housing
# Import de la función de evaluación del error del modelo
from sklearn.metrics import root_mean_squared_error
# Import de la función de separado de conjuntos de entrenamiento/test
from sklearn.model_selection import train_test_split
# Import del modelo de regresión de perceptron multicapa
from sklearn.neural_network import MLPRegressor
# Import de la función para crear una pipeline
from sklearn.pipeline import make_pipeline
# Import del transformer StandardScaler
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
housing.data, housing.target, random_state=42)

# Definición de un MLP con 3 capas ocultas de 50 neuronas cada una,
# con "early stopping" para escoger un modelo y asi evitar caer en un mínimo local en el entrenamiento y 
# ahorrar epocas. Además, esto hace que se guarde un 10% del conjunto de entrenamiento para validar.
# Verbose true para ver en la terminal el progreso del entrenamiento
mlp_reg = MLPRegressor(hidden_layer_sizes=[50, 50, 50], early_stopping=True,
verbose=True, random_state=42)

# Definición de la pipeline, primero escalado de los parametros y luego entrenamiento del MLP
pipeline = make_pipeline(StandardScaler(), mlp_reg)
# Ejecución de la pipeline
pipeline.fit(X_train, y_train)
# Guardado del modelo entrenado
dump(mlp_reg, "modelos/mlpreg.joblib")

# MLPRegressor de Scikit-learn utiliza R² por defecto para la evaluación del modelo
# Como podemos ver en el log de entrenamiento, la validación ha ido incrementando hasta que ha llegado
# a un valle en el cual se ha abortado el entrenamiento
""" 
Validation score: 0.790473
Iteration 44, loss = 0.13194766
Validation score: 0.791439
Iteration 45, loss = 0.12960481
Validation score: 0.788517
Validation score did not improve more than tol=0.000100 for 10 consecutive epochs. Stopping.
"""

print(f"Mejor puntuación de evaluación: {mlp_reg.best_validation_score_}")

y_pred = pipeline.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
print(f"Error cuadratico: {rmse}")


