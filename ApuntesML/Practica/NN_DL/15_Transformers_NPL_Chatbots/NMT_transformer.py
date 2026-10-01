import torch
import torch.nn as nn
from positional_encoding import PositionalEmbedding

# Objetivo del script: modelo de traducción automática neural (NMT) basado en un Transformer
# encoder-decoder. Usa el nn.Transformer de PyTorch, en vez de las capas escritas a mano, y le añade
# los embeddings de entrada, la codificación posicional, las máscaras y la capa de salida.
class NmtTransformer(nn.Module):
    def __init__(self, vocab_size, max_length, embed_dim=512, pad_id=0,
    num_heads=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        # Embedding: vocab_size categorías de entrada (un id por token), embed_dim salidas. Convierte
        # [B, L] ids enteros en [B, L, embed_dim]. padding_idx=pad_id hace que el token de relleno
        # tenga un vector nulo que no se actualiza durante el entrenamiento.
        self.pos_embed = PositionalEmbedding(max_length, embed_dim, dropout)
        # Codificación posicional: se suma a los embeddings, porque las capas de atención no saben
        # en qué posición está cada token. max_length es la longitud máxima de las secuencias.
        # Se usa la misma para el origen y el destino.
        self.transformer = nn.Transformer(
        embed_dim, num_heads, num_encoder_layers=num_layers,
        num_decoder_layers=num_layers, batch_first=True)
        # Transformer completo (encoder + decoder): d_model = embed_dim, num_heads cabezas y
        # num_layers capas en cada lado (6 en el paper). batch_first=True: los tensores son
        # [B, L, d_model] y no [L, B, d_model]. El dropout de este módulo es el de por defecto
        # (0.1): el parámetro dropout de aquí solo llega a pos_embed.
        self.output = nn.Linear(embed_dim, vocab_size)
        # Capa lineal final: un logit por cada token del vocabulario en cada posición del destino.

    def forward(self, pair):
        # pair agrupa los ids y las máscaras del origen (src) y del destino (tgt). Sus máscaras
        # valen 1 en los tokens reales y 0 en el relleno.
        src_embeds = self.pos_embed(self.embed(pair.src_token_ids))
        tgt_embeds = self.pos_embed(self.embed(pair.tgt_token_ids))
        # [B, L] ids -> [B, L, embed_dim] con embedding, y luego se suma la posición. Ambas frases
        # (la entrada del encoder y la del decoder) pasan por el mismo embedding.
        src_pad_mask = ~pair.src_mask.bool()
        tgt_pad_mask = ~pair.tgt_mask.bool()
        # PyTorch espera True en las posiciones que hay que ignorar (relleno), así que se invierte
        # la máscara. Forma [B, L].
        size = [pair.tgt_token_ids.size(1)] * 2
        full_mask = torch.full(size, True, device=tgt_pad_mask.device)
        causal_mask = torch.triu(full_mask, diagonal=1)
        # Máscara causal [Lt, Lt]: triu con diagonal=1 deja True por encima de la diagonal, es decir,
        # las posiciones futuras. Así cada token destino solo puede atender a sí mismo y a los
        # anteriores (para que no copie la respuesta durante el entrenamiento).
        out_decoder = self.transformer(src_embeds, tgt_embeds,
        src_key_padding_mask=src_pad_mask,
        memory_key_padding_mask=src_pad_mask,
        tgt_mask=causal_mask, tgt_is_causal=True,
        tgt_key_padding_mask=tgt_pad_mask)
        # Llamada al Transformer: src_key_padding_mask ignora el relleno del origen en la
        # self-attention del encoder; memory_key_padding_mask hace lo mismo en la cross-attention del
        # decoder (la "memoria" es la salida del encoder, con la misma longitud que el origen);
        # tgt_mask es la máscara causal; tgt_is_causal=True avisa de que esa máscara es causal;
        # tgt_key_padding_mask ignora el relleno del destino. Salida: [B, Lt, embed_dim].
        return self.output(out_decoder).permute(0, 2, 1)
        # La capa lineal produce [B, Lt, vocab_size], pero CrossEntropyLoss espera las clases en la
        # segunda dimensión, así que se permuta a [B, vocab_size, Lt].