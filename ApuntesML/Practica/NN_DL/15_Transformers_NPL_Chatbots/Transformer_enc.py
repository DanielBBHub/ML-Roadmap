# Objetivo del script: una capa del encoder del Transformer original ("Attention Is All You Need").
# Tiene dos subcapas, self-attention multicabeza y red feedforward, y cada una va envuelta en una
# conexión residual + LayerNorm. El encoder completo apila N de estas capas (N = 6 en el paper).
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout)
        # Self-attention: cada token actualiza su representación atendiendo a todos los tokens de
        # la misma frase. d_model entradas y salidas, nhead cabezas de d_model / nhead dimensiones.
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        # Primera capa del feedforward: expande cada token de d_model a dim_feedforward
        # (2048 en el paper). Se le aplica una ReLU en el forward.
        self.dropout = nn.Dropout(dropout)
        # Dropout compartido por todas las subcapas, como regularización.
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # Segunda capa del feedforward, sin activación: vuelve a d_model para poder sumar la
        # salida a la conexión residual (las formas tienen que coincidir).
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Layer normalization: normaliza cada token por separado sobre sus d_model
        # características (no sobre el batch). Estabiliza el entrenamiento. Una para cada subcapa.
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # src: [B, L, d_model]. src_mask: [L, L] bloquea pares de posiciones concretos (casi
        # nunca se usa en el encoder). src_key_padding_mask: [B, L], True en los tokens de relleno
        # (padding) para que nadie les preste atención.
        attn, _ = self.self_attn(src, src, src, attn_mask=src_mask,
        key_padding_mask=src_key_padding_mask)
        # Self-attention: query, key y value son el mismo tensor. Se descartan los pesos (_).
        # attn tiene forma [B, L, d_model].
        Z = self.norm1(src + self.dropout(attn))
        # Conexión residual (skip connection) + LayerNorm: la entrada se suma a la salida de la
        # atención, lo que facilita que fluya el gradiente. Es el esquema "post-norm" del paper.
        ff = self.dropout(self.linear2(self.dropout(self.linear1(Z).relu())))
        # Feedforward aplicado a cada token por separado: [B, L, d_model] -> [B, L, dim_feedforward]
        # -> ReLU -> [B, L, d_model]. Los tokens solo se mezclan en la atención, no aquí.
        return self.norm2(Z + ff)
        # Segunda residual + LayerNorm. La salida mantiene la forma de la entrada, [B, L, d_model],
        # lo que permite apilar capas.