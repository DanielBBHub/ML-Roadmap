from sklearn.datasets import fetch_openml
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
# Import de la función para crear una pipeline
from sklearn.pipeline import make_pipeline
from joblib import dump, load

# Fetch del dataset de imagenes de ropa MINST
fashion_mnist = fetch_openml(name="Fashion-MNIST", as_frame=False)
# Conversion de las etiquetas 0...9 de str a int
targets = fashion_mnist.target.astype(int) 

# División en conjunto de entrenamiento y test, de 70k, 10k a test y 60k a entrenamiento
X_train, y_train = fashion_mnist.data[:60_000], targets[:60_000]
X_test, y_test = fashion_mnist.data[60_000:], targets[60_000:]

# Comprobación de la primera img, reescalada a 28x28 ya que la entrada es un array de 784 valores de intensidad
""" X_sample = X_train[0].reshape(28, 28) # first image in the training set
plt.imshow(X_sample, cmap="binary")
plt.show() """

# Definicion en str de lo que representan las etiquetas en int
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
"Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Definicion de un MLPClassifier, con 2 capas ocultas con 300 y 100 neuronas cada una
mlp_clf = MLPClassifier(hidden_layer_sizes=[300, 100], verbose=True,
early_stopping=True, random_state=42)
# Definicion de la pipeline con un Transformer para las normalizar 
# la intensidad de los pixeles de las imagenes y el clasificador
# Se utiliza MinMaxScaler debido a la "importancia" de ciertos píxeles; 
# los de los bordes serán todos blancos y puede que distorsionen el resultado
# de la transformacion
pipeline = make_pipeline(MinMaxScaler(), mlp_clf)

pipeline.fit(X_train, y_train)
# Guardado del modelo entrenado
dump(mlp_clf, "modelos/mlpclass.joblib")

accuracy = pipeline.score(X_test, y_test)

print(f"Mejor puntuación de evaluación: {mlp_clf.best_validation_score_}")

X_new = X_test[:15]
y_pred = pipeline.predict(X_new)
y_proba = mlp_clf.predict_proba(X_new)
# La prediccion 12/15 es incorrecta. Cuando comparamos las predicciones y sus probabilidades podemos ver que esta seguro
# al 100% de las predicciones, aun siendo incorrectas. Esto nos demuestra que no puede tomarse a rajatabla las probabilidades 
# estimadas de las predicciones, sobretodo si hay overfitting o se ha entrenado durante demasiadas epocas
for pred,prob in zip(y_pred, y_proba):
    print(f"Clase predecida: {pred} ||||||||| Probabilidad de la clase: {prob[pred]}")
