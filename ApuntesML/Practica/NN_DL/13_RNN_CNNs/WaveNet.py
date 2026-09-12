import torch.nn as nn
import torch.nn.functional as F

# WaveNet simplificado: apilar convoluciones 1D causales y dilatadas en vez de
# usar celulas recurrentes
""" WaveNet ataja el problema de la memoria a largo plazo de una manera muy
distinta a la RNN/GRU/LSTM: en vez de mantener un estado que se va
actualizando paso a paso, apila varias capas convolucionales 1D cuyo "dilation
rate" (la separacion entre los pasos temporales que combina cada kernel) va
creciendo, normalmente duplicandose en cada capa (1, 2, 4, 8, ...). Esto hace
que el "campo receptivo" (cuantos pasos de la secuencia original influyen en
cada salida) crezca exponencialmente con el numero de capas en vez de
linealmente, permitiendo capturar patrones muy largos con relativamente pocas
capas y sin tener que propagar un estado paso a paso """

class CausalConv1d(nn.Conv1d):
    """Version "causal" de nn.Conv1d: una convolucion normal con dilatacion
    mezclaria, en cada posicion de salida, pasos pasados y futuros de la
    entrada (y ademas acortaria la secuencia). Aqui, en vez de eso, rellenamos
    la entrada por la izquierda con ceros antes de aplicar la convolucion, de
    forma que cada salida en la posicion t solo depende de la entrada en los
    pasos <= t (nunca de pasos futuros) y la secuencia de salida conserva
    exactamente la misma longitud que la de entrada. Al heredar de nn.Conv1d
    y solo sobreescribir forward(), seguimos teniendo disponibles kernel_size,
    dilation, etc., tal y como los configuremos al crear la capa"""
    def forward(self, X):
        # El campo receptivo de un kernel dilatado de tamaño kernel_size con
        # una tasa de dilatacion dilation abarca (kernel_size - 1) * dilation
        # pasos hacia atras ademas del propio paso actual; ese es justo el
        # numero de ceros que hay que añadir a la izquierda para que la
        # primera salida ya tenga todo el contexto que necesita y no se pierda
        # ningun paso por el camino
        padding = (self.kernel_size[0] - 1) * self.dilation[0]
        # F.pad(X, (padding, 0)) añade "padding" ceros a la izquierda del
        # ultimo eje de X (el eje temporal) y 0 a la derecha, es decir, solo
        # rellena por el pasado, nunca por el futuro
        X = F.pad(X, (padding, 0))
        # Con la entrada ya rellenada, delegamos en la convolucion normal de
        # nn.Conv1d (super().forward()), que ahora producira una secuencia de
        # salida de la misma longitud que la de entrada original
        return super().forward(X)

class WavenetModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        layers = []
        # Se apilan 8 capas convolucionales causales con dilatacion 1, 2, 4, 8,
        # 1, 2, 4, 8 (el patron 1, 2, 4, 8 se repite dos veces). kernel_size=2
        # en todas ellas: cada capa combina el paso actual con un unico paso
        # anterior, pero ese paso anterior esta cada vez mas lejos en el
        # tiempo segun crece la dilatacion, por lo que el campo receptivo
        # conjunto de las 8 capas crece muy rapido (de forma exponencial) sin
        # necesitar kernels grandes ni muchisimas capas
        for dilation in (1, 2, 4, 8) * 2:
            conv = CausalConv1d(input_size, hidden_size, kernel_size=2, dilation=dilation)
            # Cada convolucion va seguida de una activacion ReLU, igual que en
            # cualquier CNN
            layers += [conv, nn.ReLU()]
            # A partir de la primera capa, la entrada de las siguientes ya no
            # tiene input_size canales sino hidden_size (el numero de filtros
            # que produce cada capa convolucional)
            input_size = hidden_size
        # Encadenamos las 8 parejas (convolucion, ReLU) en un unico modulo
        # secuencial
        self.convs = nn.Sequential(*layers)
        # Capa de salida aplicada a cada paso temporal (igual que en un modelo
        # seq2seq), para pasar de hidden_size canales a la prediccion final
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        # X llega con forma [batch_size, window_length, input_size]. Igual que
        # con nn.Conv1d en el modelo de downsampling, hay que colocar el eje
        # temporal al final y el de canales en el medio antes de pasar por las
        # convoluciones
        Z = X.permute(0, 2, 1)
        Z = self.convs(Z)
        # Deshacemos el permute para volver a dejar el eje temporal en el
        # medio, forma [batch_size, window_length, hidden_size], ya que
        # gracias al padding causal la longitud de la secuencia no ha cambiado
        # en ningun momento
        Z = Z.permute(0, 2, 1)
        # La capa Linear se aplica de forma independiente en cada paso
        # temporal, dando una prediccion por cada uno de ellos, con forma
        # final [batch_size, window_length, output_size]
        return self.output(Z)

torch.manual_seed(42)
wavenet_model = WavenetModel(input_size=5, hidden_size=32, output_size=14)
wavenet_model = wavenet_model.to(device)

""" Gracias al padding causal, cada capa convolucional produce una secuencia
de salida exactamente de la misma longitud que su entrada, asi que el modelo
completo no necesita recortar ni recortar (downsample) los targets como
pasaba con la convolucion con stride del modelo anterior: se puede entrenar
directamente con los mismos targets seq2seq (secuencias completas de 112
dias), reutilizando los mismos data loaders que ya se construyeron para el
modelo seq2seq basado en GRU/LSTM (seq_train_loader, seq_eval_loader, etc.) """
