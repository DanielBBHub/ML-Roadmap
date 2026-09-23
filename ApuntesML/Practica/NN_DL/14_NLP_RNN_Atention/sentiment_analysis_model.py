import torch
import torch.nn as nn
import transformers
from torch.nn.utils.rnn import pack_padded_sequence

class SentimentAnalysisModel(nn.Module):
    # Modelo secuencia-a-vector: a partir de una reseña tokenizada (secuencia),
    # produce un único logit que indica si el sentimiento es positivo o negativo
    # (vector de tamaño 1). A diferencia de ShakespeareModel, aquí no importa la
    # salida en cada paso temporal, solo la predicción final tras leer toda la reseña.
    def __init__(self, vocab_size, n_layers=2, embed_dim=128, hidden_dim=64,
    pad_id=0, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim,
        padding_idx=pad_id)
        # Embedding: vocab_size categorías de entrada (una por token del
        # vocabulario), embed_dim salidas. Convierte [batch, ventana] en
        # [batch, ventana, embed_dim]. padding_idx=pad_id fija el vector del
        # token de padding a ceros y evita que reciba gradiente durante el
        # entrenamiento, para que el padding (necesario porque las reseñas
        # tienen longitudes distintas) no distorsione la pérdida.
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=n_layers,
        batch_first=True, dropout=dropout)
        # GRU: embed_dim entradas (deben coincidir con la salida del embedding),
        # hidden_dim salidas (tamaño del estado oculto), n_layers capas apiladas
        # con dropout entre ellas para regularizar. batch_first=True porque si no,
        # la capa asume que la dimensión de batch va después de la temporal.
        self.output = nn.Linear(hidden_dim, 1)
        # Lineal: hidden_dim entradas (deben coincidir con el estado oculto de la
        # GRU), una única salida porque es clasificación binaria: el logit
        # resultante será positivo para reseñas positivas y negativo para negativas
        # (se interpretaría con una sigmoide/BCEWithLogitsLoss, no con softmax).

    def forward(self, encodings):
        # encodings es un objeto BatchEncoding (el que devuelve el tokenizador de
        # Hugging Face), no un tensor plano: además de los IDs de token incluye
        # información como la máscara de atención. Aquí solo se usan los IDs,
        # que ya vienen con padding y truncado aplicados por el tokenizador.
        embeddings = self.embed(encodings["input_ids"])
        _outputs, hidden_states = self.gru(embeddings)
        # Se descarta outputs (las salidas en cada paso temporal, forma
        # [batch, ventana, hidden_dim]) porque para clasificar una reseña completa
        # solo hace falta el resumen final. hidden_states[-1] es el estado oculto
        # final de la última capa GRU, forma [batch, hidden_dim] — equivalente a
        # outputs[:, -1], que también se podría haber usado.
        return self.output(hidden_states[-1])
        # Tensor 2D [batch, 1]: un logit por reseña del lote.

""" 
Aun que este modelo si se entrenase con algo como BCEWithLogitsLoss pudiese llegar a algo como 85% de precision, un problema que tiene este es que no se esta ignorando completamente los 0s del padding,
con lo que si una reseña acaba teniendo mucho padding, el modulo GRU tendra que procesarlos y al acabar la reseña puede que haya "olvidado" de que iba la reseña. Para evitar esto se puede utilizar
una secuencia empaquetada en vez de un tensor normal. Esta secuencia empaquetada es una estructura de datos diseñada para representar eficientemente un lote de secuencias de longitudes variables. Se
puede utilizar la funcion pad_packed_sequence() cuando se quiera convertir una secuencia empaquetada a un tensor con padding. Hay que tener en cuenta que esta funcion espera que las secuencias esten
ordenadas de mayor a menor longitud, si no definir enforce_sorted=False. Ademas, esta espera que la dimension del tiempo vaya primero, si no definir batch_first=True.

Las capas recurrentes de pytorch soportan secuencias empaquetadas; procesan eficientemente las secuencias, parando al final de cada una.
"""


class SentimentAnalysisModelPacked(nn.Module):
    def __init__(self, vocab_size, n_layers=2, embed_dim=128, hidden_dim=64,
    pad_id=0, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=n_layers,
        batch_first=True, dropout=dropout)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, encodings):
        embeddings = self.embed(encodings["input_ids"])
        lengths = encodings["attention_mask"].sum(dim=-1).cpu()
        # pack_padded_sequence "empaqueta" el batch: en lugar de procesar
        # también los tokens de padding paso a paso (aunque su embedding sea
        # cero), construye un PackedSequence que solo contiene los tokens
        # reales de cada reseña, indicando lengths (número de tokens no
        # padding por secuencia, calculado a partir de attention_mask) para
        # saber dónde corta cada una. enforce_sorted=False permite pasar las
        # longitudes en cualquier orden (si no, la GRU exige que el batch
        # venga ordenado de mayor a menor longitud).
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False)
        _packed_outputs, hidden_states = self.gru(packed_embeddings)
        # La GRU acepta un PackedSequence igual que un tensor normal, pero al
        # no iterar sobre el padding es más eficiente, y sobre todo cambia el
        # resultado de hidden_states: ahora es exactamente el estado oculto
        # tras el último token real de cada reseña, sin haber "seguido leyendo"
        # padding después. Con la versión sin empaquetar, aunque el padding se
        # embeba como ceros, la GRU seguía dando pasos con esa entrada cero y
        # podía "olvidar" parte de la información de la reseña si había mucho
        # padding; aquí queda evitado por completo.
        return self.output(hidden_states[-1])


class SentimentAnalysisModelBidirectional(nn.Module):
    def __init__(self, vocab_size, n_layers=2, embed_dim=128, hidden_dim=64,
    pad_id=0, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=n_layers,
        batch_first=True, dropout=dropout, bidirectional=True)
        # bidirectional=True añade, por cada capa, una segunda GRU que recorre la
        # secuencia en sentido inverso (del último token al primero), de modo que
        # cada posición queda resumida con contexto tanto de lo anterior como de
        # lo posterior. Esto duplica el número de estados ocultos que devuelve la
        # capa: pasan de forma [n_layers, batch, hidden_dim] a
        # [2 * n_layers, batch, hidden_dim], alternando el estado de la pasada
        # hacia delante y el de la pasada hacia atrás por cada capa apilada.
        self.output = nn.Linear(2 * hidden_dim, 1)
        # Lineal: 2 * hidden_dim entradas, el doble que en las versiones
        # anteriores, porque hay que concatenar el estado oculto final de la
        # pasada hacia delante con el de la pasada hacia atrás de la última capa.
        # Sigue teniendo una única salida por seguir siendo clasificación binaria.

    def forward(self, encodings):
        embeddings = self.embed(encodings["input_ids"])
        lengths = encodings["attention_mask"].sum(dim=-1).cpu()
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False)
        _packed_outputs, hidden_states = self.gru(packed_embeddings)
        # hidden_states tiene forma [2 * n_layers, batch, hidden_dim]: por cada
        # capa apilada guarda primero el estado de la pasada hacia delante y a
        # continuación el de la pasada hacia atrás. Solo interesan los de la
        # última capa, es decir, las dos últimas posiciones de la primera
        # dimensión: hidden_states[-2:], de forma [2, batch, hidden_dim].
        n_dims = self.output.in_features
        top_states = hidden_states[-2:].permute(1, 0, 2).reshape(-1, n_dims)
        # permute(1, 0, 2) pone el batch en primer lugar: [batch, 2, hidden_dim].
        # reshape(-1, n_dims) aplana las dos últimas dimensiones (2 y hidden_dim)
        # en una sola de tamaño 2 * hidden_dim, concatenando así, para cada
        # reseña, el estado final hacia delante con el estado final hacia atrás
        # en un único vector que ya encaja con las entradas de self.output.
        return self.output(top_states)


class SentimentAnalysisModelPreEmbeds(nn.Module):
    def __init__(self, pretrained_embeddings, n_layers=2, hidden_dim=64,
    dropout=0.2):
        super().__init__()
        weights = pretrained_embeddings.weight.data
        # from_pretrained construye la capa de embedding copiando una matriz de
        # pesos ya entrenada (aquí, la de un modelo BERT) en vez de inicializarla
        # aleatoriamente. freeze=True impide que esos pesos reciban gradiente, así
        # que el modelo reaprovecha tal cual las representaciones que BERT ya
        # aprendió sobre un corpus mucho mayor que IMDb, en lugar de tener que
        # aprender los embeddings desde cero con las (relativamente pocas)
        # reseñas disponibles.
        self.embed = nn.Embedding.from_pretrained(weights, freeze=True)
        embed_dim = weights.shape[-1]
        # El tamaño del embedding ya no es un hiperparámetro propio del modelo:
        # viene impuesto por el modelo preentrenado (weights tiene forma
        # [vocab_size, embed_dim]), así que se lee de sus pesos en vez de
        # recibirlo como argumento.
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=n_layers,
        batch_first=True, dropout=dropout)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, encodings):
        # Igual que en SentimentAnalysisModelPacked: se empaqueta la secuencia
        # usando las longitudes reales (a partir de attention_mask) para que la
        # GRU ignore el padding por completo.
        embeddings = self.embed(encodings["input_ids"])
        lengths = encodings["attention_mask"].sum(dim=-1).cpu()
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False)
        _packed_outputs, hidden_states = self.gru(packed_embeddings)
        return self.output(hidden_states[-1])

class SentimentAnalysisModelBert(nn.Module):
    # A diferencia de SentimentAnalysisModelPreEmbeds, que solo reutilizaba la
    # matriz de embedding estática de BERT (un vector fijo por token, siempre el
    # mismo independientemente del contexto), aquí se usa el propio modelo BERT
    # como codificador: cada token se representa con un embedding contextual,
    # que sí varía según las palabras que lo rodean (por ejemplo, "banco" tendría
    # un vector distinto en "me senté en el banco" y en "fui al banco a sacar dinero").
    def __init__(self, n_layers=2, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.bert = transformers.AutoModel.from_pretrained("bert-base-uncased")
        # Se carga el modelo BERT preentrenado completo (no solo su capa de
        # embedding), con todos sus pesos ya entrenados sobre un corpus masivo.
        for param in self.bert.parameters():
            param.requires_grad = False
        # BERT tiene decenas de millones de parámetros: reentrenarlo entero con
        # el (relativamente pequeño) conjunto de IMDb sería caro y arriesgaría
        # sobreescribir lo que ya aprendió con un corpus mucho mayor. Al congelar
        # sus parámetros (requires_grad=False) se usa como un extractor de
        # características fijo, y solo se entrenan la GRU y la capa lineal de
        # clasificación que van encima.
        embed_dim = self.bert.config.hidden_size
        # El tamaño de la representación ya no es un hiperparámetro propio: lo
        # fija la arquitectura de BERT (hidden_size, 768 para bert-base), así que
        # se lee de su configuración en vez de pasarlo como argumento.
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=n_layers,
        batch_first=True, dropout=dropout)
        # GRU: embed_dim entradas (deben coincidir con hidden_size de BERT),
        # hidden_dim salidas (tamaño del estado oculto), n_layers capas apiladas,
        # igual que en el resto de variantes del modelo.
        self.output = nn.Linear(hidden_dim, 1)
        # Lineal: hidden_dim entradas, una única salida por seguir siendo
        # clasificación binaria, igual que en el resto de variantes del modelo.

    def forward(self, encodings):
        with torch.no_grad():
            contextualized_embeddings = self.bert(**encodings).last_hidden_state
        # torch.no_grad() evita que se construya el grafo de autograd para el
        # paso por BERT: como esta congelado (requires_grad=False en todos sus
        # parametros) ese grafo nunca se usaria para retropropagar, asi que
        # calcularlo solo gastaria memoria y computo de mas.
        # Se pasan los IDs y la attention_mask del BatchEncoding directamente al
        # modelo BERT (**encodings). last_hidden_state es la salida de su última
        # capa, con forma [batch, ventana, embed_dim]: un vector contextual por
        # token, el equivalente de "embeddings" en las clases anteriores pero ya
        # calculado por BERT en vez de por una capa nn.Embedding propia.
        lengths = encodings["attention_mask"].sum(dim=1).cpu()
        # Igual que en las clases anteriores: longitud real de cada secuencia a
        # partir de la attention_mask, para poder empaquetarla e ignorar el padding.
        # pack_padded_sequence exige que lengths viva en CPU, aunque el resto del
        # tensor esté en GPU, de ahí el .cpu().
        packed = pack_padded_sequence(contextualized_embeddings, lengths,
        batch_first=True, enforce_sorted=False)
        # Empaqueta contextualized_embeddings igual que en SentimentAnalysisModelPacked,
        # para que la GRU no procese los pasos de padding.
        _outputs, hidden_states = self.gru(packed)
        return self.output(hidden_states[-1])
        # Igual que en SentimentAnalysisModelPacked: el estado oculto final de la
        # última capa GRU (ya sin contaminación de padding) es el vector que se
        # usa para clasificar la reseña completa.