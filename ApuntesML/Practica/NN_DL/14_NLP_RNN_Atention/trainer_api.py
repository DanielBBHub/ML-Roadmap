
# La API de entrenamiento

""" 
La api de entrenamiento ayuda a ajustar un modelo sobre tu propio dataset con poco codigo boilerplate, ademas puede reducir los checkpoints de los modelos 
en entrenamiento, aplicar early stoppings, deistribuir los calculos en gpus, registrar metricas, manejar los paddings, lotes, barajar datos y mas

Esta API trabaja directamente con objetos Dataset, pero espera que los datasets contengan texto tokenizado, con lo que tendremos que preprocesar el texto. Podemos
hacerlo utilizando la funcion map()
"""
bert_tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")


from transformers import BertForSequenceClassification
torch.manual_seed(42)
bert_for_binary_clf = BertForSequenceClassification.from_pretrained(
"bert-base-uncased", num_labels=2, dtype=torch.float16).to(device)

def tokenize_batch(batch):
    return bert_tokenizer(batch["text"], truncation=True, max_length=200)

imdb_dataset = load_dataset("stanfordnlp/imdb")
split = imdb_dataset["train"].train_test_split(train_size=0.8, seed=42)
imdb_train_set, imdb_valid_set = split["train"], split["test"]
imdb_test_set = imdb_dataset["test"]

# Definir batched=True acelera el proceso de preprocesado pasando lotes de reseñas a la funcion tokenize_batch
tok_imdb_train_set = imdb_train_set.map(tokenize_batch, batched=True)
tok_imdb_valid_set = imdb_valid_set.map(tokenize_batch, batched=True)
tok_imdb_test_set = imdb_test_set.map(tokenize_batch, batched=True)

# Para evaluar el modelo podemos simplemente utilizar esta funcion que recibe un objeto con dos atributos: label_ids y predictions
def compute_accuracy(pred):
    return {"accuracy": (pred.label_ids == pred.predictions.argmax(-1)).mean()}

# Ahora especificaremos los argumentos de entrenamiento
from transformers import TrainingArguments
train_args = TrainingArguments(
    output_dir="my_imdb_model", num_train_epochs=2,
    per_device_train_batch_size=128, per_device_eval_batch_size=128,
    eval_strategy="epoch", logging_strategy="epoch", save_strategy="epoch",
    load_best_model_at_end=True, metric_for_best_model="accuracy",
    report_to="none")

# Finalmente creamos un objeto Trainer y pasamos el modelo con los argumentos de entrenamiento, los conjuntos de entrenamiento y validacion
# la funcion de evaluacion y un Data collator que se ocupara del padding. Una vez hecho, solo hay que llamar al metodo train().

from transformers import DataCollatorWithPadding, Trainer
trainer = Trainer(
bert_for_binary_clf, train_args, train_dataset=tok_imdb_train_set,
eval_dataset=tok_imdb_valid_set, compute_metrics=compute_accuracy,
data_collator=DataCollatorWithPadding(bert_tokenizer))
train_output = trainer.train()