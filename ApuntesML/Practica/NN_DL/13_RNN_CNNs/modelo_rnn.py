# Implementando en base a principios RNN
class SimpleRnnModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """El constructor recibe tres argumentos: el tamaño de entrada, el tamaño
        oculto y el tamaño de salida. En nuestro caso input_size = 1 porque
        trabajamos con una serie de tiempo univariable. hidden_size es el numero
        de neuronas recurrentes (32 en este ejemplo), un hiperparametro ajustable.
        output_size = 1 porque solo predecimos un unico valor futuro"""
        super().__init__()
        self.hidden_size = hidden_size
        # La celula de memoria se usa una vez por paso temporal: es un modulo
        # secuencial compuesto de una capa Linear y la activacion tanh. Se usa
        # tanh porque suele ser mas estable que otras activaciones dentro de RNNs
        self.memory_cell = nn.Sequential(
        nn.Linear(input_size + hidden_size, hidden_size),
        nn.Tanh()
        )
        # Capa lineal que toma el ultimo estado oculto y produce la salida final.
        # Es necesaria porque el estado oculto tiene una dimension por neurona
        # recurrente (32) mientras el objetivo tiene una unica dimension, y ademas
        # tanh solo saca valores entre -1 y +1, mientras los valores a predecir
        # ocasionalmente superan +1
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        # forward() recibe lotes de entrada del data loader, con forma
        # [batch_size, window_length, dimensionality], dimensionality = 1 aqui
        batch_size, window_length, dimensionality = X.shape
        # Intercambiamos las dos primeras dimensiones para poder iterar paso a
        # paso: en cada iteracion X_t tiene forma [batch_size, dimensionality]
        X_time_first = X.transpose(0, 1)
        # El estado oculto H se inicializa a 0: un valor por neurona recurrente
        # y por instancia del lote, forma [batch_size, hidden_size]
        H = torch.zeros(batch_size, self.hidden_size, device=X.device)
        for X_t in X_time_first:
            # Concatenamos la entrada actual X_t y el estado oculto H a lo largo
            # de la primera dimension, dando XH de forma
            # [batch_size, input_size + hidden_size]
            XH = torch.cat((X_t, H), dim=1)
            # Pasamos XH por la celula de memoria para obtener el nuevo estado oculto
            H = self.memory_cell(XH)
        # Tras el bucle, H es el estado oculto final; lo pasamos por la capa de
        # salida para obtener la prediccion, de forma [batch_size, output_size]
        return self.output(H)
    
# Utilizando el modulo nn.RNN
# Para hacer profunda esta red solo hay que definir una cantidad de capas >1
class RNNWithModule(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """En el constructor creamos un modulo nn.RNN en vez de construir la celula
        de memoria a mano. Se especifican input_size y hidden_size igual que antes.
        batch_first=True porque nuestros lotes de entrada tienen la dimension del
        batch primero; si no se indicase, nn.RNN asumiria que la dimension temporal
        va primero (forma [window_length, batch_size, dimensionality] en vez de
        [batch_size, window_length, dimensionality])"""
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        # La capa de salida se crea exactamente igual que en la implementacion anterior
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        # Pasamos el lote de entrada directamente al modulo nn.RNN, que se encarga
        # de todo internamente: inicializa el estado oculto a 0 y procesa cada paso
        # temporal con una celula de memoria simple (Linear + tanh por defecto),
        # igual que haciamos a mano en la implementacion anterior
        #
        # nn.RNN devuelve dos valores:
        # - outputs: las salidas de la capa recurrente superior en cada paso
        #   temporal, de forma [batch_size, window_length, hidden_size] (con
        #   batch_first=True; si no, las dos primeras dimensiones irian al reves).
        #   Con una unica capa recurrente estas salidas son simplemente los
        #   estados ocultos de esa capa en cada paso, pero nn.RNN admite varias
        #   capas recurrentes apiladas (num_layers > 1) para hacer la red profunda
        # - last_state: el estado oculto de cada capa recurrente tras el ultimo
        #   paso temporal, de forma [num_layers, batch_size, hidden_size]. Con
        #   una sola capa, la primera dimension vale 1
        outputs, last_state = self.rnn(X)
        # Tomamos la ultima salida (que coincide con el ultimo estado de la capa
        # recurrente superior) y la pasamos por la capa de salida Linear
        return self.output(outputs[:, -1])