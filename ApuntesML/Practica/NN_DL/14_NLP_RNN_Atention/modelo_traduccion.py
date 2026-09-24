import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Modelo de traducción automática neuronal (NMT) inglés → español con
# arquitectura encoder-decoder (seq2seq) basada en GRU, sin atención: el
# encoder resume la frase inglesa en sus estados ocultos finales y el decoder
# genera la frase española condicionado por ese resumen.
class NmtModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, pad_id=0, hidden_dim=512,
    n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        # Embedding compartido por encoder y decoder: vocab_size entradas (10 000,
        # el vocabulario BPE común a inglés y español, porque se usa un único
        # tokenizador) y embed_dim=512 salidas. [batch, long] → [batch, long, 512].
        # padding_idx=pad_id: el vector del token <pad> (id 0) se fija a ceros y
        # no recibe gradiente, así el relleno no aprende nada.
        self.encoder = nn.GRU(embed_dim, hidden_dim, num_layers=n_layers,
        batch_first=True)
        # Encoder: GRU de 2 capas apiladas, 512 entradas (salida del embedding) y
        # 512 unidades de estado oculto. Lee la frase en inglés y la condensa en
        # su estado oculto final (el "vector de contexto").
        self.decoder = nn.GRU(embed_dim, hidden_dim, num_layers=n_layers,
        batch_first=True)
        # Decoder: misma forma que el encoder (mismo n_layers y hidden_dim), y
        # tiene que ser así porque su estado inicial será el estado final del
        # encoder, de forma [n_layers, batch, hidden_dim].
        self.output = nn.Linear(hidden_dim, vocab_size)
        # Capa de salida: 512 → vocab_size, un logit por cada token posible del
        # vocabulario en cada paso temporal (clasificación sobre 10 000 clases).

    def forward(self, pair):
        src_embeddings = self.embed(pair.src_token_ids)
        # Frase inglesa: [batch, src_len] → [batch, src_len, 512].
        tgt_embeddings = self.embed(pair.tgt_token_ids)
        # Entrada del decoder: frase española desplazada a la derecha
        # ("<s> ..." sin el último token, ver nmt_collate_fn) →
        # [batch, tgt_len, 512]. Es teacher forcing: en entrenamiento el decoder
        # recibe el token correcto anterior, no su propia predicción.
        src_lengths = pair.src_mask.sum(dim=1)
        # Longitud real de cada frase fuente: la máscara vale 1 en tokens reales
        # y 0 en <pad>, así que sumar por filas da el nº de tokens reales. [batch]
        src_packed = pack_padded_sequence(
        src_embeddings, lengths=src_lengths.cpu(),
        batch_first=True, enforce_sorted=False)
        # Empaqueta las secuencias para que la GRU se detenga en el último token
        # real de cada frase y no procese el relleno; si no, el estado final
        # quedaría "contaminado" por los <pad>. lengths debe estar en CPU, y
        # enforce_sorted=False evita tener que ordenar el batch por longitud.
        _, hidden_states = self.encoder(src_packed)
        # Solo interesan los estados ocultos finales de TODAS las capas:
        # [n_layers, batch, hidden_dim] = [2, batch, 512]. Las salidas por paso
        # se descartan (un modelo con atención sí las usaría).
        outputs, _ = self.decoder(tgt_embeddings, hidden_states)
        # El decoder arranca con el estado del encoder como estado inicial (cada
        # capa recibe el de su capa homóloga) y procesa la frase española.
        # outputs: [batch, tgt_len, 512], uno por paso temporal de la última capa.
        return self.output(outputs).permute(0, 2, 1)
        # Lineal: [batch, tgt_len, 512] → [batch, tgt_len, vocab_size]. Se permuta
        # a [batch, vocab_size, tgt_len] porque CrossEntropyLoss y Accuracy
        # esperan la dimensión de clases (logits) en segunda posición.


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


# Mismo encoder-decoder que NmtModel, pero con atención: en cada paso el decoder consulta las salidas del encoder
# para TODOS los tokens de la frase inglesa, en vez de depender solo del estado final como único vector de contexto.
class NmtModelWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, pad_id=0, hidden_dim=512,
    n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.encoder = nn.GRU(embed_dim, hidden_dim, num_layers=n_layers,
        batch_first=True)
        self.decoder = nn.GRU(embed_dim, hidden_dim, num_layers=n_layers,
        batch_first=True)
        # Embedding, encoder y decoder idénticos a NmtModel. Encoder y decoder deben tener el mismo hidden_dim tanto
        # para pasar el estado final de uno al otro como para que el producto escalar de la atención tenga sentido.
        self.output = nn.Linear(2 * hidden_dim, vocab_size)
        # Capa de salida: 2 * hidden_dim = 1024 entradas porque recibe la concatenación de la salida de la atención
        # (512) y la salida del decoder (512); vocab_size salidas, un logit por token del vocabulario.

    def forward(self, pair):
        src_embeddings = self.embed(pair.src_token_ids) # [batch, src_len, 512]
        tgt_embeddings = self.embed(pair.tgt_token_ids) # [batch, tgt_len, 512]
        src_lengths = pair.src_mask.sum(dim=1)
        src_packed = pack_padded_sequence(
        src_embeddings, lengths=src_lengths.cpu(),
        batch_first=True, enforce_sorted=False)
        encoder_outputs_packed, hidden_states = self.encoder(src_packed)
        # Ahora sí se conservan las salidas del encoder en cada paso (una por token de la frase fuente): son las
        # keys y values de la atención. hidden_states sigue siendo [n_layers, batch, 512] e inicializa el decoder.
        decoder_outputs, _ = self.decoder(tgt_embeddings, hidden_states)
        # [batch, tgt_len, 512]: una salida por paso del decoder, que harán de queries.
        encoder_outputs, _ = pad_packed_sequence(encoder_outputs_packed,
        batch_first=True)
        # Deshace el empaquetado para volver a un tensor normal [batch, src_len, 512], con ceros en las posiciones
        # de padding (que la GRU no llegó a procesar).
        attn_output = attention(query=decoder_outputs, key=encoder_outputs,
        value=encoder_outputs)
        # [batch, tgt_len, 512]: para cada paso del decoder, resumen de la frase inglesa centrado en los tokens más
        # relevantes para predecir el siguiente token español.
        combined_output = torch.cat((attn_output, decoder_outputs), dim=-1)
        # [batch, tgt_len, 1024]: se combina "qué dice la frase fuente" (atención) con "qué llevo traducido" (decoder).
        return self.output(combined_output).permute(0, 2, 1)
        # [batch, tgt_len, vocab_size] → [batch, vocab_size, tgt_len], clases en segunda posición para CrossEntropyLoss.
