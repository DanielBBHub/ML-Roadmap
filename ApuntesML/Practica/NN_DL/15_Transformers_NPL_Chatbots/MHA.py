# Objetivo del script: implementar desde cero la capa de atención multicabeza (MHA) del
# Transformer. Aplica la atención de producto punto escalado, softmax(QK^T / sqrt(d_k)) V,
# en h cabezas en paralelo y luego concatena sus salidas y las mezcla con una capa lineal.
class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.h = num_heads
        # h: número de cabezas de atención.
        self.d = embed_dim // num_heads
        # d: dimensiones por cabeza (d_k = d_v). embed_dim debe ser divisible por num_heads;
        # p. ej. 512 / 8 = 64, de modo que h × d = embed_dim.
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        # Proyecciones lineales de Q, K y V (los W_Q, W_K, W_V que se aprenden): embed_dim
        # entradas y embed_dim salidas. Una sola capa por tensor sirve para las h cabezas a la
        # vez: sus embed_dim salidas se reparten después en h bloques de d (split_heads), y
        # cada bloque es la proyección propia de una cabeza.
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        # Capa lineal final: mezcla las salidas concatenadas de todas las cabezas
        # [B, Lq, h × d] -> [B, Lq, embed_dim], que hasta ahora han trabajado por separado.
        self.dropout = nn.Dropout(dropout)
        # Dropout sobre los pesos de atención (no sobre las entradas), como regularización.

    def split_heads(self, X):
        # [B, L, h × d] -> [B, L, h, d]: reparte la última dimensión en h cabezas de d
        # dimensiones cada una.
        # .transpose(1, 2) -> [B, h, L, d]: pone h junto a B. El operador @ solo actúa sobre
        # las dos últimas dimensiones, así que así calcula todas las cabezas y todo el batch
        # de una vez.
        return X.view(X.size(0), X.size(1), self.h, self.d).transpose(1, 2)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        q = self.split_heads(self.q_proj(query)) # (B, h, Lq, d)
        k = self.split_heads(self.k_proj(key)) # (B, h, Lk, d)
        v = self.split_heads(self.v_proj(value)) # (B, h, Lv, d) with Lv=Lk
        # En self-attention query, key y value son el mismo tensor; en cross-attention, la query
        # viene del decoder y key/value de la salida del encoder (por eso Lq y Lk pueden diferir).
        scores = q @ k.transpose(2, 3) / self.d**0.5 # (B, h, Lq, Lk)
        # Puntuaciones de similitud query-clave: [Lq, d] @ [d, Lk] -> [Lq, Lk] por cabeza.
        # Se divide entre sqrt(d_k) para evitar que el softmax se sature (gradientes diminutos).
        # Enmascarado: se pone -inf en las puntuaciones que hay que ignorar, para que el softmax les
        # dé peso 0. attn_mask [Lq, Lk] bloquea pares concretos (p. ej. los tokens futuros en la MHA
        # enmascarada del decoder); key_padding_mask [B, Lk] bloquea los tokens de relleno.
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, -torch.inf) # (B, h, Lq, Lk)
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2) # (B, 1, 1, Lk)
            scores = scores.masked_fill(mask, -torch.inf) # (B, h, Lq, Lk)
        weights = scores.softmax(dim=-1) # (B, h, Lq, Lk)
        # Softmax por filas: cada token de la query reparte un peso total de 1 entre los tokens
        # de la clave.
        Z = self.dropout(weights) @ v # (B, h, Lq, d)
        # Media ponderada de los values: [Lq, Lk] @ [Lk, d] -> [Lq, d]. Cada fila es la
        # representación contextualizada de un token de la query en esa cabeza.
        Z = Z.transpose(1, 2) # (B, Lq, h, d)
        # Se devuelve h junto a d, para poder concatenarlas.
        Z = Z.reshape(Z.size(0), Z.size(1), self.h * self.d) # (B, Lq, h × d)
        # Concatenación de las salidas de las h cabezas a lo largo de las características
        # (reshape y no view porque tras el transpose el tensor no es contiguo en memoria).
        return (self.out_proj(Z), weights) # (B, Lq, h × d)
        # Devuelve la salida mezclada y los pesos de atención (por si se quieren inspeccionar).