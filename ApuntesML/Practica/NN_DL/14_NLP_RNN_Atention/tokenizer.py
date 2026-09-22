from pathlib import Path
import urllib.request
import torch.nn.functional as F
import torch
# Descarga del 25% de las obras de Sheakespeare
def download_shakespeare_text():
    path = Path("datasets/shakespeare/shakespeare.txt")
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://homl.info/shakespeare"
        urllib.request.urlretrieve(url, path)
    return path.read_text()

shakespeare_text = download_shakespeare_text()
#print(shakespeare_text[:80])

""" 
Las redes neuronales trabajan con números, con lo que tendremos que convertir este texto en numeros, normalmente se hace separando el texto en "tokens", como palabras o 
caracteres y asignandoles un ID entero a cada posible token.
"""

# Cada uno de los caracteres utilizados en esta obra será parte del vocabulario de tokens
vocab = sorted(set(shakespeare_text.lower()))
#print(f"\nVocabulario completo: {"".join(vocab)}")

char_to_id = {char: index for index, char in enumerate(vocab)}
id_to_char = {index: char for index, char in enumerate(vocab)}
#print(f"\nEquivalencia de caracter 'a' a id: {char_to_id["a"]}")
#print(f"Equivalencia de id '13' a caracter: {id_to_char[13]}")


import torch
# Funciones auxiliares para crear tensores de tokesn en base a texto y viceversa
def encode_text(text):
    return torch.tensor([char_to_id[char] for char in text.lower()])

def decode_text(char_ids):
    return "".join([id_to_char[char_id.item()] for char_id in char_ids])

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


#encoded = encode_text("Hello, world!")
#print("\nPalabra a codificar/descodificar: 'Hello, world!'")
#print(f"Tensor de tokens: {encoded}")
#print(f"Palabra descodificada: {decode_text(encoded)}")

