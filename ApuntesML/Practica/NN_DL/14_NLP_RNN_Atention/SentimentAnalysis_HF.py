from datasets import load_dataset
import tokenizers
import torch

""" 
El analisis de sentimientos es una de las aplicaciones más comunes de la clasificacion de texto mediante PNL, en este caso vamos a hacerlo sobre el dataset de IMDb, que es 
el equivalente al MINST para la calsificacion de imagenes; el "hola mundo" del dominio
"""

## Importar y separar el conjunto de datos en entrenamiento, validación y test
imdb_dataset = load_dataset("stanfordnlp/imdb")
split = imdb_dataset["train"].train_test_split(train_size=0.8, seed=42)
imdb_train_set, imdb_valid_set = split["train"], split["test"]
imdb_test_set = imdb_dataset["test"]

## Podemos ver que los valores de "text" y "label" son la reseña y la clasificacion de la pelicula
imdb_train_set[1]["text"]
imdb_train_set[1]["label"]
imdb_train_set[16]["text"]
imdb_train_set[16]["label"]

# Tokenizacion utilizando la libreria de Hugging Face

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

#print(f"\nTexto codificado en tokens{bpe_encoding.tokens}")
bpe_token_ids = bpe_encoding.ids
#print(f"Ids del texto codificado en tokens{bpe_token_ids}")
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

# Construyendo y entrenando un modelo de analisis de sentimientos

from tokenizer import download_shakespeare_text
from Token_dataset import CharDataset
from torch.utils.data import DataLoader

window_length = 50
batch_size = 512 
shakespeare_text = download_shakespeare_text()
train_set = CharDataset(shakespeare_text[:1_000_000], window_length)
valid_set = CharDataset(shakespeare_text[1_000_000:1_060_000], window_length)
test_set = CharDataset(shakespeare_text[1_060_000:], window_length)

""" 
El modelo se ha de entrenar utilizando lotes de reseñas tokenizadas, pero estos datasets no se han tokenizado. El proceso es tan simple como manejar la tokenizacion en los dataloaders, utilizando 
el argumento collate_fn; el DataLoader llamara a esta funcion para cada lote, pasandole una lista de muestras del conjunto de datos, para que nuestra funcion reciba este lote, tokenize las reseñas, las trunque
y meta padding y devuelva un BatchEncoding que contenga tensores con los token ID y las mascaras de atencion, asi como otro tensor con las etiquetas.
"""
import transformers
bert_tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
def collate_fn(batch, tokenizer=bert_tokenizer):
    reviews = [review["text"] for review in batch]
    labels = [[review["label"]] for review in batch]
    encodings = tokenizer(reviews, padding=True, truncation=True,
    max_length=200, return_tensors="pt")
    labels = torch.tensor(labels, dtype=torch.float32)
    return encodings, labels

batch_size = 256
imdb_train_loader = DataLoader(imdb_train_set, batch_size=batch_size,
collate_fn=collate_fn, shuffle=True)
imdb_valid_loader = DataLoader(imdb_valid_set, batch_size=batch_size,collate_fn=collate_fn)
imdb_test_loader = DataLoader(imdb_test_set, batch_size=batch_size,
collate_fn=collate_fn)

## Entrenamos los dos modelos (con padding normal y con secuencias empaquetadas) con los mismos
## hiperparametros para comparar como afecta el padding al desempeño

from sentiment_analysis_model import SentimentAnalysisModel, SentimentAnalysisModelPacked, SentimentAnalysisModelPreEmbeds, SentimentAnalysisModelBert
from ModelUtl.Train import entrenar_nn, eval_entrenamiento
from ModelUtl.SNL import saveModel
from torchmetrics.classification import BinaryAccuracy
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
bert_model = transformers.AutoModel.from_pretrained("bert-base-uncased")

vocab_size = bert_tokenizer.vocab_size
pad_id = bert_tokenizer.pad_token_id
n_iter = 40

torch.manual_seed(42)
model_padded = SentimentAnalysisModel(vocab_size, pad_id=pad_id).to(device)
optimizador_padded = torch.optim.Adam(model_padded.parameters(), lr=1e-3)
perdida_padded = nn.BCEWithLogitsLoss()
metrica_padded = BinaryAccuracy().to(device)
mejor_modelo_padded = entrenar_nn(model_padded, optimizador_padded, perdida_padded, metrica_padded,
imdb_train_loader, imdb_valid_loader, n_iter, device)

torch.manual_seed(42)
model_packed = SentimentAnalysisModelPacked(vocab_size, pad_id=pad_id).to(device)
optimizador_packed = torch.optim.Adam(model_packed.parameters(), lr=1e-3)
perdida_packed = nn.BCEWithLogitsLoss()
metrica_packed = BinaryAccuracy().to(device)
mejor_modelo_packed = entrenar_nn(model_packed, optimizador_packed, perdida_packed, metrica_packed,
imdb_train_loader, imdb_valid_loader, n_iter, device)

torch.manual_seed(42)
model_preembeds = SentimentAnalysisModelPreEmbeds(bert_model.embeddings.word_embeddings).to(device)
optimizador_preembeds = torch.optim.Adam(model_preembeds.parameters(), lr=1e-3)
perdida_preembeds = nn.BCEWithLogitsLoss()
metrica_preembeds = BinaryAccuracy().to(device)
mejor_modelo_preembeds = entrenar_nn(model_preembeds, optimizador_preembeds, perdida_preembeds, metrica_preembeds,
imdb_train_loader, imdb_valid_loader, n_iter, device)

torch.manual_seed(42)
model_bert = SentimentAnalysisModelBert().to(device)
# Solo la GRU y la capa lineal tienen requires_grad=True (self.bert se congela
# dentro de la propia clase), asi que se filtran los parametros del optimizador
# para que no incluya los de BERT: evita gastar memoria y computo manteniendo
# estado del optimizador para pesos que nunca se van a actualizar.
optimizador_bert = torch.optim.Adam(
    (p for p in model_bert.parameters() if p.requires_grad), lr=1e-3)
perdida_bert = nn.BCEWithLogitsLoss()
metrica_bert = BinaryAccuracy().to(device)
mejor_modelo_bert = entrenar_nn(model_bert, optimizador_bert, perdida_bert, metrica_bert,
imdb_train_loader, imdb_valid_loader, n_iter, device)

## Cargamos los mejores pesos de cada uno y comparamos la precision final en validacion
model_padded.load_state_dict(mejor_modelo_padded)
precision_padded = eval_entrenamiento(model_padded, imdb_valid_loader, metrica_padded, device)

model_packed.load_state_dict(mejor_modelo_packed)
precision_packed = eval_entrenamiento(model_packed, imdb_valid_loader, metrica_packed, device)
saveModel(model_packed, "sentiment_packed", {"vocab_size": vocab_size, "pad_id": pad_id})

model_preembeds.load_state_dict(mejor_modelo_preembeds)
precision_preembeds = eval_entrenamiento(model_preembeds, imdb_valid_loader, metrica_preembeds, device)
saveModel(model_preembeds, "sentiment_preembeds", {"n_layers": 2, "hidden_dim": 64, "dropout": 0.2})

model_bert.load_state_dict(mejor_modelo_bert)
precision_bert = eval_entrenamiento(model_bert, imdb_valid_loader, metrica_bert, device)
saveModel(model_bert, "sentiment_bert", {"n_layers": 2, "hidden_dim": 64, "dropout": 0.2})

print(f"Precision validacion (padding sin empaquetar): {precision_padded.item():.4f}")
print(f"Precision validacion (secuencias empaquetadas): {precision_packed.item():.4f}")
print(f"Precision validacion (embeddings preentrenados de BERT): {precision_preembeds.item():.4f}")
print(f"Precision validacion (BERT congelado + GRU): {precision_bert.item():.4f}")