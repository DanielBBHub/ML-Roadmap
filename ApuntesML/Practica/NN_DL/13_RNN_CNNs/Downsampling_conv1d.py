import torch.nn as nn

# Combinando una capa convolucional 1D con una GRU para reducir la longitud
# efectiva de la secuencia (downsampling)
""" Una de las limitaciones de las RNN (incluso con celulas LSTM o GRU) es que
su memoria a corto plazo sigue degradandose cuando las secuencias son muy
largas: cuantos mas pasos temporales tiene que recorrer la informacion, mas
dificil es que sobreviva intacta. Una forma de aliviar este problema es
acortar la secuencia que ve la parte recurrente de la red antes de que
empiece a procesarla, resumiendo grupos de pasos temporales consecutivos en
uno solo.

Para eso se puede usar una capa convolucional 1D (nn.Conv1d) como primera capa
del modelo: igual que una convolucion 2D desliza un kernel sobre una imagen
combinando pixeles vecinos, una convolucion 1D desliza un kernel sobre el eje
temporal combinando pasos vecinos. Si ademas se usa un stride mayor que 1 (en
este caso kernel_size=4 y stride=2), el kernel se salta pasos al deslizarse, y
la capa produce una secuencia de salida mas corta que la de entrada (aqui,
aproximadamente la mitad de larga): esto es un downsampling de la secuencia.
Tras esta capa convolucional, una GRU procesa la secuencia ya acortada, con
menos pasos temporales por los que propagar su estado, y finalmente una capa
densa (Linear) produce la prediccion en cada paso de esa secuencia reducida """
class DownsamplingModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # kernel_size=4 hace que cada salida de la convolucion resuma 4 pasos
        # temporales consecutivos de la entrada, y stride=2 hace que el kernel
        # avance de 2 en 2 (en vez de de 1 en 1), solapando parcialmente esos
        # bloques de 4 y produciendo, a la salida, aproximadamente la mitad de
        # pasos temporales que a la entrada. input_size es el numero de
        # canales de entrada (features de cada paso) y hidden_size el numero
        # de filtros/canales de salida de la convolucion
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=4, stride=2)
        # La GRU recibe la secuencia ya resumida y acortada por la
        # convolucion (hidden_size canales en cada paso) y mantiene su propio
        # estado oculto de tamaño hidden_size. batch_first=True porque
        # nuestros lotes tienen la dimension del batch primero
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        # Capa de salida aplicada a cada paso temporal de la secuencia que
        # devuelve la GRU (igual que en un modelo seq2seq), para producir la
        # prediccion final a partir del estado oculto de cada paso
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        # X llega con forma [batch_size, window_length, input_size] (el eje
        # temporal en la posicion central, como espera nn.GRU con
        # batch_first=True). Sin embargo, nn.Conv1d espera el eje de canales
        # en la posicion central y el eje temporal al final: forma
        # [batch_size, input_size, window_length]. Por eso hay que
        # intercambiar (permute) las dos ultimas dimensiones antes de la
        # convolucion, tratando el tiempo como si fuese un eje espacial
        Z = X.permute(0, 2, 1) # treat time as a spatial dimension
        # La convolucion reduce la longitud temporal (por el stride=2) y
        # cambia el numero de canales de input_size a hidden_size; la salida
        # tiene forma [batch_size, hidden_size, nueva_longitud]
        Z = self.conv(Z)
        # Deshacemos el permute para volver a dejar el eje temporal en el
        # medio, forma [batch_size, nueva_longitud, hidden_size], que es lo
        # que espera nn.GRU con batch_first=True
        Z = Z.permute(0, 2, 1) # swap back time & features dimensions
        # Aplicamos una no linealidad tras la convolucion, igual que se haria
        # tras cualquier capa convolucional de una CNN
        Z = torch.relu(Z)
        # La GRU procesa la secuencia ya acortada y devuelve sus salidas en
        # cada paso temporal (forma [batch_size, nueva_longitud, hidden_size]);
        # el segundo valor devuelto (el estado oculto final) no se necesita
        # aqui, por eso se descarta con _states
        Z, _states = self.gru(Z)
        # La capa Linear se aplica de forma independiente a cada paso
        # temporal de las salidas de la GRU (igual que en un modelo seq2seq),
        # dando una prediccion por cada uno de esos pasos ya acortados, con
        # forma final [batch_size, nueva_longitud, output_size]
        return self.linear(Z)

""" Como la convolucion acorta el eje temporal, los targets usados para
entrenar este modelo tambien hay que acortarlos y desplazarlos para que
correspondan exactamente con los pasos que produce la convolucion (por
ejemplo, saltandose los primeros pasos que el kernel necesita para producir su
primera salida y tomando despues uno de cada dos, a juego con el stride). Esta
combinacion de una convolucion 1D para downsampling seguida de una capa
recurrente es una forma habitual de ayudar a una RNN a detectar patrones a mas
largo plazo, ya que reduce el numero de pasos que su estado tiene que
atravesar sin tener que renunciar a toda la resolucion temporal de golpe """
