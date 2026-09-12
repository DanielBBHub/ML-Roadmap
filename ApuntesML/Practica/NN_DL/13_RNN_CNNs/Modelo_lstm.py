import torch.nn as nn
# Implementando una LSTM (Long Short-Term Memory)
""" Una celula LSTM es una mejora de la celula recurrente simple, pensada para
que la red pueda reconocer y conservar patrones a largo plazo sin sufrir tanto
el problema de que los gradientes se desvanezcan al propagarse muchos pasos
hacia atras en el tiempo (backprop en el tiempo). Para ello separa la memoria
en dos estados: un estado a largo plazo C (long-term state) y un estado a
corto plazo H (short-term state, que coincide con la salida de la celula en
cada paso), y controla el flujo de informacion entre ambos mediante puertas
(gates) """
class LstmModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """El constructor recibe los mismos tres argumentos que en la RNN simple:
        tamaño de entrada, tamaño oculto (numero de neuronas/unidades LSTM) y
        tamaño de salida"""
        super().__init__()
        self.hidden_size = hidden_size
        # nn.LSTMCell implementa una unica celula LSTM que se ejecuta un paso
        # temporal a la vez. Internamente calcula, a partir de la entrada X_t y
        # de los estados anteriores (H, C), las activaciones de sus tres
        # puertas y de la propuesta de nuevo contenido:
        #   - puerta de olvido (forget gate): decide que parte del estado a
        #     largo plazo C se descarta
        #   - puerta de entrada (input gate): decide que parte de la propuesta
        #     de nuevo contenido se añade a C
        #   - puerta de salida (output gate): decide que parte del estado a
        #     largo plazo (ya actualizado y pasado por una tanh) se expone
        #     como estado a corto plazo H, que es tambien la salida de la
        #     celula en ese paso
        # Las tres puertas usan activacion sigmoide (salidas entre 0 y 1, que
        # funcionan como "cuanto dejar pasar"), mientras que la propuesta de
        # nuevo contenido usa tanh, igual que la celula recurrente simple
        self.memory_cell = nn.LSTMCell(input_size, hidden_size)
        # Igual que en la RNN simple, hace falta una capa de salida porque el
        # estado oculto H tiene una dimension por unidad LSTM (hidden_size)
        # mientras el objetivo a predecir tiene su propia dimension, y ademas
        # H pasa por tanh en algun punto, limitando su rango
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        # forward() recibe lotes de entrada del data loader, con forma
        # [batch_size, window_length, dimensionality]
        batch_size, window_length, dimensionality = X.shape
        # Intercambiamos las dos primeras dimensiones para poder iterar paso a
        # paso: en cada iteracion X_t tiene forma [batch_size, dimensionality]
        X_time_first = X.transpose(0, 1)
        # A diferencia de la RNN simple, aqui hay que mantener dos estados en
        # vez de uno: H (estado a corto plazo/salida) y C (estado a largo
        # plazo), ambos inicializados a 0 y de forma [batch_size, hidden_size]
        H = torch.zeros(batch_size, self.hidden_size, device=X.device)
        C = torch.zeros(batch_size, self.hidden_size, device=X.device)
        for X_t in X_time_first:
            # A diferencia de la celula recurrente simple (que solo concatena
            # X_t y H y aplica una unica capa Linear + tanh), aqui la celula
            # recibe X_t junto con la tupla de ambos estados (H, C) y devuelve
            # los estados actualizados: primero calcula internamente el nuevo
            # estado a largo plazo combinando lo que la puerta de olvido deja
            # pasar del C anterior con lo que la puerta de entrada deja pasar
            # de la propuesta de nuevo contenido, y despues obtiene el nuevo H
            # aplicando la puerta de salida sobre una version "tanh" de ese C
            H, C = self.memory_cell(X_t, (H, C))
        # Tras el bucle, H contiene el estado a corto plazo (salida) del
        # ultimo paso; al igual que en la RNN simple, se pasa por la capa de
        # salida para obtener la prediccion, de forma [batch_size, output_size].
        # C (el estado a largo plazo final) no se usa aqui, pero es el que ha
        # permitido que la informacion relevante se propague por la secuencia
        # sin degradarse tanto como en una celula recurrente simple
        return self.output(H)

""" Al separar la memoria a largo plazo de la salida a corto plazo y regular su
actualizacion mediante puertas entrenables, la LSTM puede aprender que
informacion merece la pena conservar durante muchos pasos y cual conviene
olvidar, lo que la hace mucho mas capaz que una RNN simple a la hora de
capturar patrones de largo plazo en series o secuencias largas """
