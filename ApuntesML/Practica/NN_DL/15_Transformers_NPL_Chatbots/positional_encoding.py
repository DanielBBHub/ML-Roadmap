import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEmbedding(nn.Module):
    def __init__(self, max_length, embed_dim, dropout=0.1):
        super().__init__()
        # Una manera simple de implementar este componente es utilizando capas de Embedding, aunque 
        # tambien se pueden utilizar nn.Parameter para guardar la matriz de incrustacion (inicializada de manera
        # aleatoria, de ahi el * 0.02) y añadir las primeras L filas a la entrada 
        # (L siendo la máxima longitud de la secuencia)
        self.pos_embed = nn.Parameter(torch.randn(max_length, embed_dim) * 0.02)
        # Se añade el dropout para reducir el riesgo de overfitting
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, X):
        # Se recogen los tokens embedidos desde 0 a longitud de secuencia (self.pos_embed[:X.size(1)]) y se añaden
        # al input que entra, para pasar la entrada por dropout y entrenar el incrustado posicional
        return self.dropout(X + self.pos_embed[:X.size(1)])