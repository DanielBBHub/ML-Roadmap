import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def attention(query, key, value): # note: dq == dk and Lk == Lv
    # Atención de producto escalar (Luong): cada query se compara con todas las keys y el resultado es una media
    # ponderada de las values, con más peso en las posiciones cuya key más se parece a la query.
    scores = query @ key.transpose(1, 2) # [B,Lq,dq] @ [B,dk,Lk] = [B, Lq, Lk]
    # Puntuación de alineamiento: producto escalar entre cada query (paso del decoder) y cada key (token de la
    # frase fuente). Por eso las dimensiones de query y key deben coincidir (dq == dk).
    weights = torch.softmax(scores, dim=-1) # [B, Lq, Lk]
    # softmax sobre la dimensión de las keys: para cada query, los pesos de atención suman 1 a lo largo de la frase fuente.
    return weights @ value # [B, Lq, Lk] @ [B, Lv, dv] = [B, Lq, dv]
    # Vector de contexto de cada query: media de las values ponderada por los pesos de atención.


class NmtModelWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, pad_id=0, hidden_dim=512,
    n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.encoder = nn.GRU(embed_dim, hidden_dim, num_layers=n_layers,
        batch_first=True)
        self.decoder = nn.GRU(embed_dim, hidden_dim, num_layers=n_layers,
        batch_first=True)
        self.output = nn.Linear(2 * hidden_dim, vocab_size)
        
    def forward(self, pair):
        src_embeddings = self.embed(pair.src_token_ids) # [batch, src_len, 512]
        tgt_embeddings = self.embed(pair.tgt_token_ids) # [batch, tgt_len, 512]
        src_lengths = pair.src_mask.sum(dim=1)
        src_packed = pack_padded_sequence(
        src_embeddings, lengths=src_lengths.cpu(),
        batch_first=True, enforce_sorted=False)
        encoder_outputs_packed, hidden_states = self.encoder(src_packed)
       
        decoder_outputs, _ = self.decoder(tgt_embeddings, hidden_states)
        encoder_outputs, _ = pad_packed_sequence(encoder_outputs_packed,
        batch_first=True)
        
        attn_output = attention(query=decoder_outputs, key=encoder_outputs,
        value=encoder_outputs)
       
        combined_output = torch.cat((attn_output, decoder_outputs), dim=-1)
        return self.output(combined_output).permute(0, 2, 1)
