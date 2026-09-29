from torch.utils.data import Dataset

class FechaDataset(Dataset):
    def __init__(self, fechas_df):
        self.fechas = fechas_df
    
    def __len__(self):
        return len(self.fechas)
    
    def __getitem__(self, idx):
        window = self.fechas[idx][0]
        target = self.fechas[idx][1]
        return window, target