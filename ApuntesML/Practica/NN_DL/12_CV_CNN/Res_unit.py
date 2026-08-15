import torch
from torch import torch.nn as nn

class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        DefaultConv2d = partial(
            nn.Conv2d, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.main_layers = nn.Sequential(
            DefaultConv2d(in_channels, out_channels, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            DefaultConv2d(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            )
        
        if stride > 1:
            self.skip_connection = nn.Sequential(
            DefaultConv2d(in_channels, out_channels, kernel_size=1,
            stride=stride, padding=0),
            nn.BatchNorm2d(out_channels),
            )
        
        else:
            self.skip_connection = nn.Identity()
    
    def forward(self, inputs):
        return F.relu(self.main_layers(inputs) + self.skip_connection(inputs))
