# Implementación de las capas convolucionales separables como módulo personalizado de pytorch
import torch
from torch import torch.nn as nn


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0):
        super().__init__()
        # Lo que la diferencia de una convolucional convencional es el argumento de "groups=in_chanels"
        # Normalmente in_chanels y out_chanels tienen que ser divisibles por groups, pero al definir
        # out_channels y groups como los canales de entrada se crea una depthwise 
        self.depthwise_conv = nn.Conv2d(
        in_channels, in_channels, kernel_size, stride=stride,
        padding=padding, groups=in_channels)
        # La otra parte de la capa es una convolucional normal pero con un kernel 1x1 y stride 1
        self.pointwise_conv = nn.Conv2d(
        in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, inputs):
        return self.pointwise_conv(self.depthwise_conv(inputs))