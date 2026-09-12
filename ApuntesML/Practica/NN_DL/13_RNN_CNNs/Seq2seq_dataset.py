from Forecast_ahead_dataset import ForecastAheadDataset

""" Heredamos de ForecastAheadDataset (que ya nos da el __len__() correcto) y
sobreescribimos __getitem__() para que el target no sea un unico vector de 14
valores, sino una secuencia de vectores de 14 valores, uno por cada paso
temporal de la ventana de entrada: en el paso 0 el target son los valores de
los pasos 1 a 14, en el paso 1 son los valores de los pasos 2 a 15, etc. Esto
convierte el modelo de sequence-to-vector a sequence-to-sequence (seq2seq):
dada una entrada de forma [batch_size, window_length, input_size], la salida
tendra forma [batch_size, window_length, output_size] """
class Seq2SeqDataset(ForecastAheadDataset):
    def __getitem__(self, idx):
        end = idx + self.window_length # 1st index after window
        window = self.series[idx : end]
        # target_period contiene, para la columna "rail" (0), todos los valores
        # desde un paso despues del inicio de la ventana hasta 14 pasos despues
        # del final de la ventana
        target_period = self.series[idx + 1 : end + 14, 0]
        # unfold(dimension=0, size=14, step=1) recorre target_period deslizando
        # una ventana de tamaño 14 de uno en uno, generando así todas las
        # ventanas de 14 valores consecutivos desplazadas un paso cada vez.
        # Ejemplo con torch.tensor([0, 1, 2, 3, 4, 5]).unfold(0, size=4, step=1):
        #   tensor([[0, 1, 2, 3],
        #           [1, 2, 3, 4],
        #           [2, 3, 4, 5]])
        target = target_period.unfold(dimension=0, size=14, step=1)
        return window, target

""" Es importante notar que el target contiene valores que tambien aparecen en
la entrada (excepto en el ultimo paso temporal). No es trampa: una RNN en cada
paso t solo conoce las entradas hasta ese paso, nunca las futuras, por lo que
sigue siendo un modelo causal """
