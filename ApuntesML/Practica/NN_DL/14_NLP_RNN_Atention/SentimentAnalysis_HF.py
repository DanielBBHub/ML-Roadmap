from datasets import load_dataset
import tokenizers

""" 
El analisis de sentimientos es una de las aplicaciones más comunes de la clasificacion de texto mediante PNL, en este caso vamos a hacerlo sobre el dataset de IMDb, que es 
el equivalente al MINST para la calsificacion de imagenes; el "hola mundo" del dominio
"""

## Importar y separar el conjunto de datos en entrenamiento, validación y test
imdb_dataset = load_dataset("imdb")
split = imdb_dataset["train"].train_test_split(train_size=0.8, seed=42)
imdb_train_set, imdb_valid_set = split["train"], split["test"]
imdb_test_set = imdb_dataset["test"]

## Podemos ver que los valores de "text" y "label" son la reseña y la clasificacion de la pelicula
imdb_train_set[1]["text"]
imdb_train_set[1]["label"]
imdb_train_set[16]["text"]
imdb_train_set[16]["label"]

# Tokenizacion utilizando la libreria de Hugging Face
""" 
En un paper de 2016 [ https://homl.info/rarewords ] se exploran varios metodos para tokenizar/destokenizar el texto a nivel inferior a las palabras. De esta manera en el caso de que 
hubiese una palabra nueva para el modelo pudiese inferir el significado (grandisimo -> grand (grande) + isimo(sufijo) ). Una de estas tecnicas se basa en BPE (bite par encoding)
la cual separa todo el conjunto de entrenamiento en caracteres individuales y en cada iteracion encuentra el par de tokens adyacentes mas comunes, añadiendolo al vocabulario hasta llegar
a un tamaño deseado. La libreria de Huggin Face implementa de manera eficiente varios algoritmos de tokenizacion, incluido el anterior.
"""
## Creamos un modelo BPE especificando un token desconocido "<unk>" que se utilizara en caso de encontrar una palabra que no aparezca
bpe_model = tokenizers.models.BPE(unk_token="<unk>")
bpe_tokenizer = tokenizers.Tokenizer(bpe_model)
## Es posible definir un pre y post procesado del texto. En este caso dividimos el dataset en los espacios
bpe_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
## Definimos tokens especiales; <unk> y el <pad>, el cual servira para cuando creemos lotes de texto con tamaños diferentes
special_tokens = ["<pad>", "<unk>"]
## Creamos el entrenador de BPE con el tamaño de vocabulario deseado y los tokens definidos
bpe_trainer = tokenizers.trainers.BpeTrainer(vocab_size=1000,
special_tokens=special_tokens)
## Definimos el conjunto de entrenamiento a base de reseñas de imdb
train_reviews = [review["text"].lower() for review in imdb_train_set]
## Corremos el entrenamiento
bpe_tokenizer.train_from_iterator(train_reviews, bpe_trainer)

some_review = "what an awesome movie!"
bpe_encoding = bpe_tokenizer.encode(some_review)

print(f"\nTexto codificado en tokens{bpe_encoding.tokens}")
bpe_token_ids = bpe_encoding.ids
print(f"Ids del texto codificado en tokens{bpe_token_ids}")
""" 
['what', 'an', 'aw', 'es', 'ome', 'movie', '!']
[303, 139, 373, 149, 240, 211, 4, 1]
El resultado de la tokenizacion es este. Se puede ver que palabras mas comunes como "what" y "movie" se han codificado enteras, mientras que
"awesome" se ha partido en cachos
"""
## El tokenizador tambien tiene un metodo get_vocab() que devuelve un diccionario mapeando cada token a su id. Tambien
## se puede utilizar el metodo token_to_id() para mapear un unico token o id_to_token() para mapear un id. De todas maneras
## el metodo mas utilizad sera convertir una lista de ids a un str
bpe_tokenizer.decode(bpe_token_ids)

## El tokenizador tambien lleva la cuenta de los offsets de cada token en la propiedad .offsets
bpe_encoding.offsets

## Tambien se puede codificar en masa lotes de texto con encode_batch()
bpe_tokenizer.encode_batch(train_reviews[:3])

""" 
En caso de que queramos convertir las codificaciones anteriores de las reseñas en un tensor nos tenemos que asegurar que todas tienen la misma cantidad de
tokens, lo que podemos conseguir haciendo que el tokenizador meta padding en las reseñas mas cortsa para igualar tamaños o que trunque las secuencias mas 
largas que N
"""
bpe_tokenizer.enable_padding(pad_id=0, pad_token="<pad>")
bpe_tokenizer.enable_truncation(max_length=500)

bpe_encodings = bpe_tokenizer.encode_batch(train_reviews[:3])
bpe_batch_ids = torch.tensor([encoding.ids for encoding in bpe_encodings])

""" 
tensor([[159, 402, 176, 246, 61, [...], 215, 156, 586, 0, 0, 0, 0],
[ 10, 138, 198, 289, 175, [...], 0, 0, 0, 0, 0, 0, 0],
[289, 15, 209, 398, 177, [...], 50, 29, 22, 17, 24, 18, 24]])

Podemos ver como la primera y segunda reseña tienen valores 0; tienen el padding definido antes. Ademas, cada objeto Encoding tiene un atributo 'attention_mask' que contiene
1s y 0s (para tokens normales y padding respectivamente), lo cual es muy util para que el modelo sepa que tokens ignorar, simplemente multiplicando las matrices de encoding
con las matrices de atencion. 

Por otro lado, a veces es interesante tener la lista de longitudes de las secuencias.
"""

attention_mask = torch.tensor([encoding.attention_mask for encoding in bpe_encodings])
lengths = attention_mask.sum(dim=-1)

""" 
En este ejemplo es visible que el tokenizador no ha tratado muy bien los espacios, partiendo "awesome" en "aw es ome" y "movie!" en "movie !" y esto es por que el pretokenizador 
Whitespace ha eliminado todos los espacios, con lo que el tokenizador no sabe donde debe poner los espacios y los coloca en medio de los tokens. Para arreglar este problema podemos 
reemplazar Whitespace por ByteLevel el cual reemplaza los espacios con un caracter especial 'Ġ' de manera que el modelo tokenizador no le pierde la pista a los espacios. En el 
caso anterior el resultado de tokenizar sería el siguiente: 'Ġwhat Ġan Ġaw es ome Ġmovie !', lo cual estaria casi perfecto, si no fuese que al remplazar los caracteres especiales por
espacios, habría uno al principio que no debería estar ahi.

Por otro lado, si el texto contuviese emoticonos tampoco sería capaz de tokenizarlos. ByteLevel permite que el modelo BPE funcione a nivel de bytes, en vez de caracters, con lo que 
el emoticono sera convertido a cuatro bytes con el encoding UTF-8, con lo que el modelo tokenizador nunca dejara de mostrar un token desconocido si su vocabulario contiene todos los 256 bytes posibles,
ya que cualquier texto puede desgranarse en sus bytes individuales. Esto se conoce como BBPE (byte-level BPE).

Otra variantes de BPR es WordPiece, introducida por Google en 2016 [ https://homl.info/wordpiece ], el cual en vez de agregar el par de tokens mas comun al vocabulario en cada iteracion, añade el
par con mayor puntuacion. Esta puntuacion sigue la siguiente ecuacion:

$$
score(AB) = \frac{frequency(AB)}{frequency(A)*frequency(B)}
$$

Esta forma de puntuar mejora la puntuacion de los tokens que se encuentran juntos pero penaliza los que, aun estando juntos, tambien se encuentran por separado en el texto. Para entrenar un tokenizador
WordPiece se puede utilizar el mismo codigo que el anterior, modificando el modelo BPE por WordPiece y BpeTrainer por WordPieceTrainer.

Otro algoritmo popular es Unigram LM, introducido en 2018 [ https://homl.info/subword ], el cual empieza con un vocabulario muy grande que contiene cada palabra, subpalabra y caracter frecuente en el
texto de entrenamiento y gradualmente va eliminadno los tokens menos utiles, hasta llegar a un tamaño de vocabulario deseado. Este algoritmo parte de la suposicion que el texto ha sido muestreado
de manera aleatoria del vocabulario, un token detras de otro y que cada token se ha muestreado independientemente de los otros.

| Característica | BBPE | WordPiece | Unigram LM |
|---|---|---|---|
| **Cómo funciona** | Fusiona los pares más frecuentes | Fusiona los pares que maximizan la verosimilitud de los datos | Elimina los tokens menos probables |
| **Ventajas** | Rápido, simple, ideal para multilingüe | Buen equilibrio entre eficiencia y calidad de los tokens | El más significativo, secuencias más cortas |
| **Desventajas** | Puede producir divisiones extrañas | Menos robusto que BBPE para multilingüe | Más lento de entrenar y tokenizar |
| **Usado por** | GPT, Llama, RoBERTa, BLOOM | BERT, DistilBERT, ELECTRA | T5, ALBERT, mBART |
"""