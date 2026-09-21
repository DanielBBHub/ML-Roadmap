import torch.nn as nn
# Modelo generador de texto carácter a carácter: dado un fragmento de texto
# codificado, predice qué carácter viene después en cada posición de la secuencia.
class ShakespeareModel(nn.Module):
    def __init__(self, vocab_size, n_layers=2, embed_dim=10, hidden_dim=128,
    dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # Embedding: vocab_size categorías de entrada (una por carácter distinto),
        # embed_dim salidas (hiperparámetro ajustable). Convierte tensores de
        # enteros [batch, ventana] en tensores densos [batch, ventana, embed_dim].
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=n_layers,
        batch_first=True, dropout=dropout)
        # GRU: embed_dim entradas (deben coincidir con la salida del embedding),
        # hidden_dim salidas (tamaño del estado oculto), n_layers capas apiladas
        # (RNN profunda) con dropout entre ellas para regularizar. batch_first=True
        # porque si no, la capa asume que la dimensión de batch va después de la
        # dimensión temporal. Variante de RNN con puertas de actualización/reinicio
        # que mitiga el gradiente desvaneciente en secuencias largas.
        self.output = nn.Linear(hidden_dim, vocab_size)
        # Lineal: debe tener vocab_size salidas porque queremos un logit por cada
        # carácter posible en cada paso temporal, para luego aplicar softmax/pérdida.

    def forward(self, X):
        embeddings = self.embed(X)
        outputs, _states = self.gru(embeddings)
        # La GRU también devuelve el estado oculto final, pero se ignora (_states):
        # cada lote se procesa de forma independiente, sin arrastrar memoria entre
        # lotes (modelo "sin estado"). outputs tiene forma [batch, ventana, hidden_dim].
        return self.output(outputs).permute(0, 2, 1)
        # La capa lineal produce [batch, ventana, vocab_size], pero CrossEntropyLoss
        # espera la dimensión de clases en segunda posición, no en la última, así
        # que se permutan las dos últimas dimensiones antes de devolver el resultado.

