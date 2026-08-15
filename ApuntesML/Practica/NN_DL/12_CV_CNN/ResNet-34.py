import torch
from torch import torch.nn as nn
import ResidualUnit

class ResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        
        layers = [
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2,
            padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        
        prev_filters = 64
        
        for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
            stride = 1 if filters == prev_filters else 2
            layers.append(ResidualUnit(prev_filters, filters, stride=stride))
            prev_filters = filters
        
        layers += [
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.LazyLinear(10),
        ]
        
        self.resnet = nn.Sequential(*layers)
    
    def forward(self, inputs):
        return self.resnet(inputs)