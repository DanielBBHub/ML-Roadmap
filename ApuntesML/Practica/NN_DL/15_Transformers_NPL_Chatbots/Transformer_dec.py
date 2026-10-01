# Objetivo del script: una capa del decoder del Transformer original. Tiene tres subcapas:
# self-attention enmascarada sobre la frase destino, cross-attention hacia la salida del encoder
# y red feedforward, cada una con conexión residual + LayerNorm. El decoder apila N capas.
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn1 = MultiheadAttention(d_model, nhead, dropout)
        # MHA enmascarada (self-attention causal): Q, K y V vienen del propio decoder, y la máscara
        # causal impide mirar a los tokens posteriores.
        self.self_attn2 = MultiheadAttention(d_model, nhead, dropout)
        # Aunque se llame self_attn2, es la cross-attention: Q viene del decoder, K y V de la salida
        # del encoder. Es una instancia distinta, con sus propias proyecciones q/k/v/out.
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # Feedforward por token: d_model -> dim_feedforward (ReLU en el forward) -> d_model.
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # Una LayerNorm por cada subcapa, que normaliza cada token sobre sus d_model características.

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
        tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # tgt: [B, Lt, d_model], la frase destino (decoder). memory: [B, Ls, d_model], la salida
        # final del encoder. tgt_mask: [Lt, Lt], la máscara causal. memory_mask: [Lt, Ls], casi
        # nunca se usa. Las *_key_padding_mask son [B, L] con True en los tokens de relleno.

        attn1, _ = self.self_attn1(tgt, tgt, tgt, attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask)
        # Self-attention enmascarada: cada token del destino solo atiende a sí mismo y a los
        # anteriores, para que el modelo no haga trampas al entrenar. Salida [B, Lt, d_model].
        Z = self.norm1(tgt + self.dropout(attn1))
        # Residual + LayerNorm.
        attn2, _ = self.self_attn2(Z, memory, memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)
        # Cross-attention: query = Z (decoder), key = value = memory (encoder). La matriz de pesos es
        # [Lt, Ls]: cada token destino reparte su atención entre los tokens de la frase origen. La
        # salida mantiene la longitud del decoder, [B, Lt, d_model]. La máscara de padding evita
        # atender a los tokens de relleno del origen.

        Z = self.norm2(Z + self.dropout(attn2))
        # Residual + LayerNorm.
        ff = self.dropout(self.linear2(self.dropout(self.linear1(Z).relu())))
        # Feedforward por token, igual que en el encoder.
        return self.norm3(Z + ff)
        # Tercera residual + LayerNorm. Forma de salida [B, Lt, d_model], igual que la de tgt.