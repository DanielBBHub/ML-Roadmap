""" Heredamos de SimpleRnnModel (self.rnn y self.output ya estan definidos en su
__init__) y solo cambiamos el forward(): en vez de aplicar self.output unicamente
a la salida del ultimo paso temporal (outputs[:, -1]), la aplicamos a las
salidas de todos los pasos temporales """
class Seq2SeqRnnModel(SimpleRnnModel):
    def forward(self, X):
        # outputs tiene forma [batch_size, window_length, hidden_size]
        outputs, last_state = self.rnn(X)
        # nn.Linear normalmente se aplica a entradas 2D [batch_size, features],
        # pero tambien funciona con entradas de mas dimensiones: se aplica de
        # forma independiente a cada paso temporal (usa torch.matmul() por
        # debajo, que soporta multiplicar arrays con mas de 2 dimensiones,
        # multiplicando entre si las matrices que corresponden a cada "posicion"
        # de las dimensiones extra, con broadcasting incluido). Por eso aqui el
        # resultado tiene forma [batch_size, window_length, output_size]: una
        # prediccion de 14 valores por cada paso temporal, no solo el ultimo
        return self.output(outputs)

""" Una alternativa a nn.Linear para esta capa de salida es nn.Conv1d con
kernel_size=1 (Conv1d(32, 14, kernel_size=1)), que da exactamente el mismo
resultado; solo habria que intercambiar las dos ultimas dimensiones de entrada
y salida, tratando el eje temporal como si fuese un eje espacial """

""" Durante el entrenamiento se usan todas las salidas del modelo (una por cada
paso temporal), lo cual hace que la funcion de perdida reciba gradientes de
error desde cada paso y no solo desde el ultimo. Esto estabiliza y acelera el
entrenamiento, y ademas el modelo ve ventanas de entrada de longitudes
efectivas variables (ya que en los primeros pasos "ve" pocos valores previos),
lo que puede reducir el sobreajuste a la longitud de ventana usada en el
entrenamiento.

Tras entrenar, solo nos interesa la salida del ultimo paso temporal (las demas
se pueden ignorar), ya que es la que contiene la prediccion de los siguientes
14 valores a partir de toda la ventana de entrada completa """
