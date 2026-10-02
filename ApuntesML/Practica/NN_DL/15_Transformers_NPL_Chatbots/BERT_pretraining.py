# Objetivo del script: preentrenar desde cero un BERT pequeño con modelado de lenguaje
# enmascarado (MLM) sobre WikiText-2, usando las librerías de Hugging Face, y probarlo
# después con una tarea fill-mask. BERT es solo el encoder del Transformer: cada token
# atiende a todos los demás (self-attention bidireccional, sin máscara causal).
from transformers import BertConfig, BertForMaskedLM, BertTokenizerFast

bert_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
# Tokenizador WordPiece de bert-base-uncased (pasa todo a minúsculas). Solo se reutiliza el
# tokenizador preentrenado, no los pesos del modelo. Su vocabulario es de 30522 tokens.
config = BertConfig( # adapt to training budget, and dataset size & complexity
vocab_size=bert_tokenizer.vocab_size, hidden_size=128, num_hidden_layers=2,
num_attention_heads=4, intermediate_size=512, max_position_embeddings=128)
# Hiperparámetros de una versión diminuta de BERT (BERT-base usa 768, 12 capas y 12 cabezas):
# - vocab_size: filas de la capa de embedding y salidas del logit final (uno por token).
# - hidden_size=128: dimensión de embedding y de cada token a lo largo de todo el encoder
#   (el embed_dim de la MHA). [batch, L] -> [batch, L, 128].
# - num_hidden_layers=2: bloques de encoder apilados (el N de la Fig. 15-3).
# - num_attention_heads=4: cabezas de la MHA; cada una trabaja con 128 / 4 = 32 dimensiones.
# - intermediate_size=512: neuronas de la capa oculta del feed forward de cada bloque
#   (4 × hidden_size, la proporción habitual).
# - max_position_embeddings=128: longitud máxima de secuencia, ya que BERT usa embeddings
#   posicionales aprendidos; debe coincidir con el max_length del tokenizado.
bert = BertForMaskedLM(config)
# Encoder BERT + cabeza de MLM (Linear hidden_size -> vocab_size) que da un logit por token
# del vocabulario en cada posición. Se crea a partir de config, con pesos aleatorios: no se
# parte de un modelo preentrenado.

from datasets import load_dataset

def tokenize(example, tokenizer=bert_tokenizer):
    return tokenizer(example["text"], truncation=True, max_length=128,

padding="max_length")
# truncation=True corta las secuencias a 128 tokens y padding="max_length" rellena las más
# cortas hasta 128, de modo que todos los ejemplos tienen forma [128]. Devuelve input_ids y
# attention_mask (0 en el relleno, para que la MHA lo ignore).
mlm_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
# WikiText-2 (versión raw, sin preprocesar): artículos de Wikipedia, con muchas líneas
# vacías o títulos; cada ejemplo es una línea de texto en la columna "text".
mlm_dataset = mlm_dataset.map(tokenize, batched=True)
# batched=True aplica tokenize a lotes de ejemplos a la vez, mucho más rápido.

from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
args = TrainingArguments(output_dir="./my_bert", num_train_epochs=5,
per_device_train_batch_size=16)
# Configuración del entrenamiento: 5 épocas, lotes de 16 secuencias [16, 128]; los pesos
# se guardan en ./my_bert. El resto de hiperparámetros toma los valores por defecto
# (optimizador AdamW, lr = 5e-5).
mlm_collator = DataCollatorForLanguageModeling(bert_tokenizer, mlm=True,
mlm_probability=0.15)
# Aquí se implementa el MLM. El collator enmascara cada lote al vuelo, así que cada época
# ve máscaras distintas: elige un 15 % de los tokens y, de ellos, el 80 % se sustituye por
# [MASK], el 10 % por un token aleatorio y el 10 % se deja igual. Crea las labels con el
# token original en las posiciones elegidas y -100 en el resto, por lo que la pérdida
# (entropía cruzada) solo se calcula sobre los tokens enmascarados.
trainer = Trainer(model=bert, args=args, train_dataset=mlm_dataset,
data_collator=mlm_collator)
# Trainer encapsula el bucle de entrenamiento (forward, pérdida, backward, optimizador,
# batching y colocación en GPU).
trainer_output = trainer.train()

import torch
from transformers import pipeline
torch.manual_seed(42)
fill_mask = pipeline("fill-mask", model=bert, tokenizer=bert_tokenizer)
# Pipeline de inferencia para MLM: tokeniza el texto, pasa por el modelo, localiza el
# [MASK] y aplica softmax sobre el vocabulario en esa posición.
top_predictions = fill_mask("The capital of [MASK] is Rome.")
# Devuelve por defecto los 5 tokens más probables, cada uno con su score, el token
# predicho (token_str) y la frase completa.
top_predictions[0]
# La predicción más probable.

""" 
Este modelo devolverá resultados pesimos, ya que tiene muy poco entrenamiento. Estos modelos requieren MUCHA información y MUCHO tiempo de entrenamiento, con lo que
habitualmente se suelen utilizar modelos preentrenados sobre información lo más parecida posible a la que vas a utilizar y hacer finetunnign sobre tu propio dataset. 
Incluso utilizando MLM.
"""