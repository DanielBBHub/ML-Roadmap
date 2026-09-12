from Dataset_serie_temporal import TimeSeriesDataset

""" En vez de predecir un unico valor futuro, esta clase prepara el dataset para
predecir los siguientes 14 valores de golpe (sequence-to-vector, pero con
output_size=14 en vez de 1). Para ello solo hay que cambiar como se construye
el target: en vez de ser el valor justo despues de la ventana, pasa a ser un
vector con los 14 valores que le siguen. Se hereda de TimeSeriesDataset y solo
se sobreescriben __len__() y __getitem__() """
class ForecastAheadDataset(TimeSeriesDataset):
    def __len__(self):
        # Como ahora necesitamos 14 valores despues de cada ventana (y no solo 1),
        # hay 14 - 1 ventanas menos disponibles que en el dataset original
        return len(self.series) - self.window_length - 14 + 1

    def __getitem__(self, idx):
        end = idx + self.window_length # 1st index after window
        window = self.series[idx : end]
        # El target ya no es un unico valor (self.series[end]) sino los 14 valores
        # que siguen a la ventana. Se usa la columna 0 porque en la serie
        # multivariable esa es la que corresponde a "rail" (la que queremos
        # predecir), aunque la ventana de entrada siga siendo multivariable
        target = self.series[end : end + 14, 0] # 0 = rail ridership
        return window, target

""" El 14 esta escrito directamente en el codigo (hardcoded); en un proyecto real
convendria convertirlo en un parametro configurable, igual que window_length """
