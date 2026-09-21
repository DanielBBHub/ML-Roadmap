from torch.utils.data import Dataset, DataLoader
from tokenizer import encode_text
class CharDataset(Dataset):
    def __init__(self, text, window_length):
        self.encoded_text = encode_text(text)
        self.window_length = window_length
    
    def __len__(self):
        return len(self.encoded_text) - self.window_length
    
    def __getitem__(self, idx):
        
        if idx >= len(self):
            raise IndexError("dataset index out of range")

        end = idx + self.window_length
        window = self.encoded_text[idx : end]
        target = self.encoded_text[idx + 1 : end + 1]
        return window, target