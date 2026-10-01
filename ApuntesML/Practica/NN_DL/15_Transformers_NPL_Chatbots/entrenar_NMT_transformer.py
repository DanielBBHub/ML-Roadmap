import sys
from pathlib import Path
from collections import namedtuple
from torch.utils.data import DataLoader
from datasets import load_dataset
import tokenizers
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy

# ModelUtl vive en la carpeta del capítulo 14; se añade al path para reutilizar entrenar_nn
sys.path.append(str(Path(__file__).resolve().parent.parent / "14_NLP_RNN_Atention"))
from ModelUtl.Train import entrenar_nn
from NMT_transformer import NmtTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

# Objetivo del script: entrenar el NmtTransformer (traducción inglés → español) igual que el
# encoder-decoder con GRU del capítulo 14, con los mismos datos, tokenizador y DataLoaders, pero con
# un Transformer reducido (4 cabezas, 2 capas por lado, embeddings de 128).

# --- Datos y tokenizador: igual que en machine_translation.py (cap. 14) ---
nmt_original_valid_set, nmt_test_set = load_dataset(
path="ageron/tatoeba_mt_train", name="eng-spa",
split=["validation", "test"])
split = nmt_original_valid_set.train_test_split(train_size=0.8, seed=42)
nmt_train_set, nmt_valid_set = split["train"], split["test"]


def train_eng_spa():
    for pair in nmt_train_set:
        yield pair["source_text"]
        yield pair["target_text"]

max_length = 256
vocab_size = 10_000

nmt_tokenizer_model = tokenizers.models.BPE(unk_token="<unk>")
nmt_tokenizer = tokenizers.Tokenizer(nmt_tokenizer_model)
nmt_tokenizer.enable_padding(pad_id=0, pad_token="<pad>")
nmt_tokenizer.enable_truncation(max_length=max_length)
nmt_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
nmt_tokenizer_trainer = tokenizers.trainers.BpeTrainer(
    vocab_size=vocab_size, special_tokens=["<pad>", "<unk>", "<s>", "</s>"])
nmt_tokenizer.train_from_iterator(train_eng_spa(), nmt_tokenizer_trainer)

fields = ["src_token_ids", "src_mask", "tgt_token_ids", "tgt_mask"]

class NmtPair(namedtuple("NmtPairBase", fields)):
    def to(self, device):
        return NmtPair(self.src_token_ids.to(device), self.src_mask.to(device),
        self.tgt_token_ids.to(device), self.tgt_mask.to(device))


def nmt_collate_fn(batch):
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

# --- Modelo y entrenamiento ---
torch.manual_seed(42)
vocab_size = nmt_tokenizer.get_vocab_size()
nmt_tr_model = NmtTransformer(vocab_size, max_length, embed_dim=128, pad_id=0,
num_heads=4, num_layers=2, dropout=0.1).to(device)

n_iter = 10
optimizador = torch.optim.NAdam(nmt_tr_model.parameters(), lr=0.001)
perdida = nn.CrossEntropyLoss(ignore_index=0) # los tokens <pad> (id 0) de las etiquetas no cuentan en la pérdida
metrica = MulticlassAccuracy(num_classes=vocab_size, ignore_index=0).to(device) # exactitud por token, sin contar el <pad>
mejor_modelo = entrenar_nn(nmt_tr_model, optimizador, perdida, metrica, nmt_train_loader, nmt_valid_loader, n_iter, device)
nmt_tr_model.load_state_dict(mejor_modelo)


def translate(model, src_text, max_length=20, pad_id=0, eos_id=3):
    tgt_text = ""
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

nmt_tr_model.eval()
print(translate(nmt_tr_model, "I like to play soccer with my friends at the beach"))
