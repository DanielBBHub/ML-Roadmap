from tokenizer import download_shakespeare_text
from Token_dataset import CharDataset
from torch.utils.data import DataLoader
from ModelUtl.Train import entrenar_nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from tokenizer import encode_text, decode_text, id_to_char
from ShakespeareModel import ShakespeareModel

window_length = 50
batch_size = 512 # reduce if your GPU cannot handle such a large batch size
shakespeare_text = download_shakespeare_text()
train_set = CharDataset(shakespeare_text[:1_000_000], window_length)
valid_set = CharDataset(shakespeare_text[1_000_000:1_060_000], window_length)
test_set = CharDataset(shakespeare_text[1_060_000:], window_length)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)

""" 
Aun teniendo ya los conjuntos de entrenamiento, validación y test, si lo utilizasemos para entrenar un modelo de ML como se ha hecho hasta ahora daría un resultado
nefasto, ya que estos entrenan para aprender que valores parecidos significan cosas parecidas y sería muy dificil quitarle el sesgo.

Una de las soluciones sería utilizar "one-hot encoding", ya que todos los vectores resultantes tendrían una separación equivalente entre ellos. Si bien no es incorrecto,
esta manera de aproximarse a la solución escala enormemente con el tamaño del vocabulario, con lo que si se hiciese con palabras, en vez de letras, explotaría en número
las dimensiones utilizadas para guardar estos vectores resultantes de "one-hot encoding". Por suerte, como estamos trabajando con redes neuronales, tenemos opciones 
mejores: los "embedings"
"""

torch.manual_seed(42) 
vocab = sorted(set(shakespeare_text.lower()))
model = ShakespeareModel(len(vocab)).to("cuda")
optimizador = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
perdida = nn.CrossEntropyLoss()
metrica = torchmetrics.Accuracy(task="multiclass", num_classes=len(vocab)).to("cuda")
entrenar_nn(model, optimizador, perdida, metrica, train_loader, valid_loader, 50, "cuda")

def next_char(model, text, temperature=1):
    encoded_text = encode_text(text).unsqueeze(dim=0).to("cuda")
    with torch.no_grad():
        Y_logits = model(encoded_text)
        Y_probas = F.softmax(Y_logits[0, :, -1] / temperature, dim=-1)
        predicted_char_id = torch.multinomial(Y_probas, num_samples=1).item()
    return id_to_char[predicted_char_id]

def extend_text(model, text, n_chars=80, temperature=1):
    for _ in range(n_chars):
        text += next_char(model, text, temperature)
    return text

model.eval()
text = "To be or not to b"
print(extend_text(model, text, temperature=0.01))
print(extend_text(model, text, temperature=0.4))
print(extend_text(model, text, temperature=100))