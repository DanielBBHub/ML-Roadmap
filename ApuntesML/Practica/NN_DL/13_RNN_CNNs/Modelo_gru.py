# Implementando una GRU (Gated Recurrent Unit)
""" La celula GRU es una version simplificada de la celula LSTM que suele
funcionar igual de bien mientras es mas rapida de entrenar. Su simplificacion
principal es que fusiona el estado a largo plazo C y el estado a corto plazo H
en un unico vector de estado H, y reduce las tres puertas de la LSTM (olvido,
entrada, salida) a solo dos:
    - puerta de actualizacion (update gate): controla, para cada componente del
      estado, si se mantiene el valor anterior de H o se sustituye por la
      propuesta de nuevo contenido. Hace a la vez el papel de las puertas de
      olvido y de entrada de la LSTM: como solo hay un "grifo", lo que no se
      actualiza con contenido nuevo se conserva automaticamente (no hace falta
      decidir a olvidar y a añadir por separado)
    - puerta de reinicio (reset gate): controla que parte del estado anterior H
      se tiene en cuenta al calcular la propuesta de nuevo contenido
No existe puerta de salida porque, al no haber un estado a largo plazo
separado, no hace falta filtrar que parte de C se expone como H: el estado
completo H es ya la salida de la celula en cada paso """
class GruModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """El constructor recibe los mismos tres argumentos que en la RNN
        simple y en la LSTM: tamaño de entrada, tamaño oculto (numero de
        unidades GRU) y tamaño de salida"""
        super().__init__()
        self.hidden_size = hidden_size
        # nn.GRUCell implementa una unica celula GRU que se ejecuta un paso
        # temporal a la vez. Recibe la entrada X_t y el estado anterior H (un
        # unico tensor, a diferencia de la tupla (H, C) de la LSTM) y devuelve
        # directamente el nuevo H, calculado internamente a partir de la
        # puerta de actualizacion, la puerta de reinicio y la propuesta de
        # nuevo contenido (las puertas usan sigmoide, la propuesta usa tanh,
        # igual que en la LSTM)
        self.memory_cell = nn.GRUCell(input_size, hidden_size)
        # Igual que en la RNN simple y en la LSTM, hace falta una capa de
        # salida porque el estado oculto H tiene una dimension por unidad GRU
        # (hidden_size) mientras el objetivo a predecir tiene su propia
        # dimension
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        # forward() recibe lotes de entrada del data loader, con forma
        # [batch_size, window_length, dimensionality]
        batch_size, window_length, dimensionality = X.shape
        # Intercambiamos las dos primeras dimensiones para poder iterar paso a
        # paso: en cada iteracion X_t tiene forma [batch_size, dimensionality]
        X_time_first = X.transpose(0, 1)
        # A diferencia de la LSTM, aqui solo hace falta mantener un unico
        # estado H (no hay C), inicializado a 0 y de forma
        # [batch_size, hidden_size]
        H = torch.zeros(batch_size, self.hidden_size, device=X.device)
        for X_t in X_time_first:
            # La celula GRU combina X_t con el H anterior y devuelve
            # directamente el H actualizado, ya con las puertas de
            # actualizacion y reinicio aplicadas por debajo
            H = self.memory_cell(X_t, H)
        # Tras el bucle, H es el estado final, que aqui hace a la vez de
        # estado a corto y a largo plazo; se pasa por la capa de salida para
        # obtener la prediccion, de forma [batch_size, output_size]
        return self.output(H)

""" Al tener menos puertas y un unico estado que mantener y propagar, la GRU es
mas ligera que la LSTM (menos parametros, menos computo por paso), y en la
practica suele alcanzar un rendimiento muy similar en la mayoria de tareas de
series temporales y secuencias """

""" PyTorch tambien ofrece nn.GRU (equivalente a nn.RNN y nn.LSTM pero con
celulas GRU), que procesa la secuencia entera de una vez sin necesidad de
escribir el bucle manual sobre los pasos temporales: basta con sustituir
nn.RNN o nn.LSTM por nn.GRU, tal y como se hizo en RNNWithModule """
