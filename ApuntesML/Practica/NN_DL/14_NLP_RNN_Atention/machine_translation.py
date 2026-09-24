from torch.utils.data import DataLoader
from datasets import load_dataset
import tokenizers
import torch
import torch.nn as nn
device = "cuda" if torch.cuda.is_available() else "cpu"


nmt_original_valid_set, nmt_test_set = load_dataset(
path="ageron/tatoeba_mt_train", name="eng-spa",
split=["validation", "test"])
split = nmt_original_valid_set.train_test_split(train_size=0.8, seed=42)
nmt_train_set, nmt_valid_set = split["train"], split["test"]


def train_eng_spa(): # a generator function to iterate over all training text
    for pair in nmt_train_set:
        yield pair["source_text"]
        yield pair["target_text"]

max_length = 256
vocab_size = 10_000

## Como los idiomas ingles y español tienen bastantes palabras parecidas no hace falta utilizar tokenizadores diferentes
nmt_tokenizer_model = tokenizers.models.BPE(unk_token="<unk>")
nmt_tokenizer = tokenizers.Tokenizer(nmt_tokenizer_model)
nmt_tokenizer.enable_padding(pad_id=0, pad_token="<pad>")
nmt_tokenizer.enable_truncation(max_length=max_length)
nmt_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
nmt_tokenizer_trainer = tokenizers.trainers.BpeTrainer(
    vocab_size=vocab_size, special_tokens=["<pad>", "<unk>", "<s>", "</s>"])
nmt_tokenizer.train_from_iterator(train_eng_spa(), nmt_tokenizer_trainer)

from collections import namedtuple
fields = ["src_token_ids", "src_mask", "tgt_token_ids", "tgt_mask"]

class NmtPair(namedtuple("NmtPairBase", fields)):
    def to(self, device):
        return NmtPair(self.src_token_ids.to(device), self.src_mask.to(device),
        self.tgt_token_ids.to(device), self.tgt_mask.to(device))


def nmt_collate_fn(batch):
    """ 
    La siguiente funcion extrae tanto el texto en ingles como en español, este ultimo con <s> al principio y </s> al final y los tokeniza. 
    Despues convierte en tensores estas entradas y etiquetas, asi como las mascaras de atencion para meterlas dentro de una named tuple que 
    hemos creado antes. 

    Devuelve la named tuple y el texto etiquetado como resultado de la ejecucion
    """
    src_texts = [pair['source_text'] for pair in batch]
    tgt_texts = [f"<s> {pair['target_text']} </s>" for pair in batch]
    src_encodings = nmt_tokenizer.encode_batch(src_texts)
    tgt_encodings = nmt_tokenizer.encode_batch(tgt_texts)
    src_token_ids = torch.tensor([enc.ids for enc in src_encodings])
    tgt_token_ids = torch.tensor([enc.ids for enc in tgt_encodings])
    src_mask = torch.tensor([enc.attention_mask for enc in src_encodings])
    tgt_mask = torch.tensor([enc.attention_mask for enc in tgt_encodings])
    inputs = NmtPair(src_token_ids, src_mask,
        tgt_token_ids[:, :-1], tgt_mask[:, :-1])
    labels = tgt_token_ids[:, 1:]
    return inputs, labels

batch_size = 32
nmt_train_loader = DataLoader(nmt_train_set, batch_size=batch_size,
collate_fn=nmt_collate_fn, shuffle=True)
nmt_valid_loader = DataLoader(nmt_valid_set, batch_size=batch_size,
collate_fn=nmt_collate_fn)
nmt_test_loader = DataLoader(nmt_test_set, batch_size=batch_size,
collate_fn=nmt_collate_fn)

from modelo_traduccion import NmtModel

torch.manual_seed(42)
vocab_size = nmt_tokenizer.get_vocab_size()
nmt_model = NmtModel(vocab_size).to(device)

from ModelUtl.Train import entrenar_nn_bleu
from torchmetrics.text import BLEUScore

n_iter = 10
optimizador = torch.optim.NAdam(nmt_model.parameters(), lr=0.001)
perdida = nn.CrossEntropyLoss(ignore_index=0) # ignore_index=0: los tokens <pad> (id 0) de las etiquetas no cuentan en la pérdida
metrica = BLEUScore()
mejor_modelo = entrenar_nn_bleu(nmt_model, optimizador, perdida, metrica, nmt_train_loader, nmt_valid_loader, n_iter, device, nmt_tokenizer)

def translate(model, src_text, max_length=20, pad_id=0, eos_id=3):
    tgt_text = ""
    token_ids = []
    for index in range(max_length):
        batch, _ = nmt_collate_fn([{"source_text": src_text,
        "target_text": tgt_text}])
        with torch.no_grad():
            Y_logits = model(batch.to(device))
            Y_token_ids = Y_logits.argmax(dim=1) # find the best token IDs
            next_token_id = Y_token_ids[0, index] # take the last token ID
        next_token = nmt_tokenizer.id_to_token(next_token_id)
        tgt_text += " " + next_token
        if next_token_id == eos_id:
            break
    return tgt_text

nmt_model.eval()
translate(nmt_model, "I like soccer.")