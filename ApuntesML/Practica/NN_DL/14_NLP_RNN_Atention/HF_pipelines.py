from transformers import pipeline

"""
La libreria de transformers provee una api muy conveniente para descargar y utilizar pipelines preentrenadas para varias tareas. Cada una dee stas contiene un modelo preentrenado
con su correspondiente pre/postprocesado.
"""
# A diferencia de todo lo anterior en esta practica (entrenar un GRU desde cero,
# reutilizar solo los embeddings de BERT, o afinar una GRU encima de BERT
# congelado), aqui no se entrena nada: se usa directamente un modelo ya
# preentrenado y afinado para analisis de sentimiento a traves de la API pipeline().

imdb_dataset = load_dataset("stanfordnlp/imdb")
split = imdb_dataset["train"].train_test_split(train_size=0.8, seed=42)
imdb_train_set, imdb_valid_set = split["train"], split["test"]
imdb_test_set = imdb_dataset["test"]

train_reviews = [review["text"].lower() for review in imdb_train_set]


model_name = "distilbert-base-uncased-finetuned-sst-2-english"
# pipeline() encapsula en un unico objeto el modelo, su tokenizador y el
# pre/postprocesado necesarios para la tarea indicada ("sentiment-analysis"); si
# no se pasara el argumento model, usaria el modelo por defecto de esa tarea. Aqui
# se especifica explicitamente distilbert-base-uncased-finetuned-sst-2-english:
# una version reducida de BERT (DistilBERT) con tokenizador sin mayusculas,
# preentrenada sobre la Wikipedia en ingles y un corpus de libros, y afinada sobre
# la tarea SST-2 (Stanford Sentiment Treebank) de clasificacion de sentimiento.
# Sobre el conjunto de validacion de IMDb este modelo concreto alcanza un 88.2%
# de precision, sin haber visto ni una reseña de IMDb durante su afinado.
# truncation=True y max_length=512 recortan las reseñas mas largas que ese limite
# de tokens antes de pasarlas al modelo, igual que se hacia con el tokenizador
# manual en las clases anteriores.
#
# El pipeline usa la GPU automaticamente si hay una disponible (con varias GPUs,
# se elige cual usar con el argumento device, el indice de la GPU); ademas, los
# modelos de transformers se cargan siempre en modo evaluacion por defecto, sin
# necesidad de llamar a .eval() como se hacia con los modelos propios.

classifier_imdb(train_reviews[:10])
# Devuelve una lista de dicts, uno por reseña, cada uno con una etiqueta
# ("POSITIVE" o "NEGATIVE") y un score: la probabilidad estimada por el modelo
# para esa etiqueta concreta, no para la clase positiva. Al ser clasificacion
# binaria, ese score nunca puede ser menor que 0.5, porque si lo fuera el modelo
# habria elegido la otra etiqueta en su lugar.