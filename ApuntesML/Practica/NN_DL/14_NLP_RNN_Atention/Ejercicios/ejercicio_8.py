"""
Entrena un modelo encoder-decoder capaz de convertir una fecha escrita en un formato a otro formato (por ejemplo, de "April 22, 2019" a "2019-04-22").
"""
from collections import namedtuple

import pandas as pd
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader

from encoder_decoder import NmtModelWithAttention
from dataset_fechas import FechaDataset
from entrenar import entrenar_nn

device = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv("dates_dataset.csv")
len_df = len(df)

train = df.values[:int(len_df * 0.7)]
valid = df.values[int(len_df * 0.7):int(len_df * 0.8)]
test = df.values[int(len_df * 0.8):]

train_dt = FechaDataset(train)
valid_dt = FechaDataset(valid)
test_dt = FechaDataset(test)

# Vocabulario a nivel de caracter, compartido entre entrada y salida (igual que
# NmtModelWithAttention, que usa un unico self.embed para src y tgt)
caracteres = sorted(set("".join(df["fecha_escrita"]) + "".join(df["fecha_iso"])))
especiales = ["<pad>", "<sos>", "<eos>"]
vocab = especiales + caracteres
char_to_id = {char: idx for idx, char in enumerate(vocab)}
pad_id, sos_id, eos_id = char_to_id["<pad>"], char_to_id["<sos>"], char_to_id["<eos>"]

def encode(texto, wrap=False):
    ids = [char_to_id[char] for char in texto]
    return [sos_id] + ids + [eos_id] if wrap else ids

def pad_batch(secuencias, pad_id):
    max_len = max(len(seq) for seq in secuencias)
    ids = torch.tensor([seq + [pad_id] * (max_len - len(seq)) for seq in secuencias])
    mask = torch.tensor([[1] * len(seq) + [0] * (max_len - len(seq)) for seq in secuencias])
    return ids, mask

fields = ["src_token_ids", "src_mask", "tgt_token_ids", "tgt_mask"]
class FechaPair(namedtuple("FechaPairBase", fields)):
    def to(self, device):
        return FechaPair(self.src_token_ids.to(device), self.src_mask.to(device),
        self.tgt_token_ids.to(device), self.tgt_mask.to(device))

def collate_fn(batch):
    src_ids = [encode(escrita) for escrita, _ in batch]
    tgt_ids = [encode(iso, wrap=True) for _, iso in batch]
    src_token_ids, src_mask = pad_batch(src_ids, pad_id)
    tgt_token_ids, tgt_mask = pad_batch(tgt_ids, pad_id)
    # Teacher forcing: el decoder recibe el target sin el ultimo token (<eos> o
    # padding) y las etiquetas son el target desplazado una posicion a la derecha
    inputs = FechaPair(src_token_ids, src_mask,
        tgt_token_ids[:, :-1], tgt_mask[:, :-1])
    labels = tgt_token_ids[:, 1:]
    return inputs, labels

batch_size = 256
fechas_train_loader = DataLoader(train_dt, batch_size=batch_size,
    collate_fn=collate_fn, shuffle=True)
fechas_valid_loader = DataLoader(valid_dt, batch_size=batch_size, collate_fn=collate_fn)
fechas_test_loader = DataLoader(test_dt, batch_size=batch_size, collate_fn=collate_fn)

torch.manual_seed(42)
perdida = nn.CrossEntropyLoss(ignore_index=pad_id)
metrica = torchmetrics.Accuracy(task="multiclass", num_classes=len(vocab),
    ignore_index=pad_id).to(device)
modelo_fechas = NmtModelWithAttention(vocab_size=len(vocab), pad_id=pad_id).to(device)
optimizador = torch.optim.Adam(modelo_fechas.parameters(), lr=1e-3, weight_decay=1e-4)
mejor_modelo = entrenar_nn(modelo_fechas, optimizador, perdida, metrica,
    fechas_train_loader, fechas_valid_loader, 20, device, paciencia=5)

modelo_fechas.load_state_dict(mejor_modelo)

def generar_fecha(modelo, texto_escrita, max_length=12):
    modelo.eval()
    src_ids = encode(texto_escrita)
    src_token_ids = torch.tensor([src_ids]).to(device)
    src_mask = torch.tensor([[1] * len(src_ids)]).to(device)
    tgt_ids = [sos_id]
    with torch.no_grad():
        for _ in range(max_length):
            tgt_token_ids = torch.tensor([tgt_ids]).to(device)
            tgt_mask = torch.tensor([[1] * len(tgt_ids)]).to(device)
            pair = FechaPair(src_token_ids, src_mask, tgt_token_ids, tgt_mask)
            logits = modelo(pair) # [1, vocab_size, tgt_len]
            siguiente_id = logits[0, :, -1].argmax().item()
            if siguiente_id == eos_id:
                break
            tgt_ids.append(siguiente_id)
    return "".join(vocab[id] for id in tgt_ids[1:]) # se descarta el <sos> inicial

print("\n--- Comprobacion sobre 3 muestras del conjunto de test ---")
for escrita, iso_esperado in test[:3]:
    iso_generado = generar_fecha(modelo_fechas, escrita)
    resultado = "OK" if iso_generado == iso_esperado else "MAL"
    print(f"Entrada: {escrita!r} -> Prediccion: {iso_generado!r} (esperado: {iso_esperado!r}) [{resultado}]")
