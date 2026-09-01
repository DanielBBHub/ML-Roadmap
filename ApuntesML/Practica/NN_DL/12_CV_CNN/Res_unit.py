import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F

class ResidualUnit(nn.Module):
    """
    Implementa un bloque residual (skip connection) para ResNet.
    Este es el componente fundamental de la arquitectura ResNet.
    
    ¿Por qué skip connections?
    ==========================
    La innovación clave de ResNet es permitir que la entrada original "salte"
    las capas convolucionales y se sume directamente a la salida. Esto permite:
    
    1. Entrenar redes MUCHO más profundas (50+, 152+ capas)
    2. Evitar el problema de vanishing gradients en redes profundas
    3. Permitir que los gradientes fluyan directamente durante backpropagation
    
    Visualización:
    ===============
    SIN skip connection (problema):
        x → [Conv] → [BN] → [ReLU] → [Conv] → [BN] → y
    
    CON skip connection (ResNet - solución):
        x → [Conv] → [BN] → [ReLU] → [Conv] → [BN] → y
        ↓_________________________________________↑
                (suma la entrada original)
        
        Resultado final: y_final = ReLU(y + x)
    
    El término "residual" viene de que la red aprende el residuo: f(x) = y - x
    En lugar de aprender la función completa, aprende solo la diferencia.
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
            in_channels: Número de canales de entrada
            out_channels: Número de canales de salida
            stride: Tamaño del paso (1=mantiene tamaño, 2=reduce a la mitad)
        """
        super().__init__()
        
        # DefaultConv2d: Crea una convolución 2D con parámetros predefinidos
        # para simplificar el código y evitar repetición
        # Parámetros fijos:
        # - kernel_size=3: Kernel pequeño (3x3)
        # - stride=1: Por defecto no cambia tamaño (se puede sobreescribir)
        # - padding=1: Mantiene las dimensiones espaciales
        # - bias=False: Sin bias porque BatchNorm lo maneja
        DefaultConv2d = partial(
            nn.Conv2d, kernel_size=3, stride=1, padding=1, bias=False)
        
        # ============================================================
        # RAMA PRINCIPAL (Main path) - Capas convolucionales
        # ============================================================
        self.main_layers = nn.Sequential(
            # CAPA 1: Primera convolución con stride variable
            # Este stride es importante: puede reducir las dimensiones espaciales
            # Ejemplo: si stride=2, la entrada de 56×56 se convierte en 28×28
            DefaultConv2d(in_channels, out_channels, stride=stride),
            nn.BatchNorm2d(out_channels),  # Normalización de lotes (acelera entrenamiento)
            nn.ReLU(),                      # Activación (introduce no-linealidad)
            
            # CAPA 2: Segunda convolución con stride=1
            # Mantiene el tamaño (no cambia dimensiones espaciales)
            DefaultConv2d(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),  # Normalización de lotes
            # ⚠️ IMPORTANTE: NO hay ReLU aquí
            # La activación ReLU se aplica DESPUÉS de sumar con la rama skip
            # Esto es crucial para el funcionamiento de las skip connections
        )
        
        # ============================================================
        # RAMA SKIP CONNECTION (conexión residual)
        # ============================================================
        # Este es el corazón de ResNet: la entrada original se suma directamente
        # a la salida de las capas convolucionales
        
        if stride > 1:
            # CASO 1: Si cambia el número de filtros o el stride es > 1
            # Necesitamos adaptar la entrada para que tenga las mismas dimensiones
            # que la salida de main_layers
            
            # Solución: Convolución 1×1 (que no extrae características espaciales,
            # solo transforma linealmente los canales)
            self.skip_connection = nn.Sequential(
                # Conv 1×1 con stride variable:
                # - Cambia el número de canales: in_channels → out_channels
                # - Reduce dimensiones espaciales si stride > 1
                #   Ejemplo: stride=2 convierte 56×56 en 28×28
                DefaultConv2d(in_channels, out_channels, kernel_size=1,
                             stride=stride, padding=0),
                nn.BatchNorm2d(out_channels),
            )
        else:
            # CASO 2: Si no cambian las dimensiones
            # Usamos Identity: una operación que no hace nada
            # La entrada pasa directamente sin modificarse
            self.skip_connection = nn.Identity()
    
    def forward(self, inputs):
        """
        Forward pass: aquí ocurre la magia de ResNet.
        
        La ecuación es: y = ReLU(main_layers(x) + skip_connection(x))
        
        Esto permite que:
        1. Los gradientes fluyan sin atenuarse (backprop más directo)
        2. La red pueda aprender tanto cambios grandes como pequeños
           (la rama principal puede decidir si cambiar la entrada o no)
        3. Las redes profundas se entrenan más fácilmente
        """
        # main_layers(inputs): Procesa la entrada a través de:
        #   [Conv] → [BN] → [ReLU] → [Conv] → [BN]
        
        # skip_connection(inputs): Devuelve:
        #   - Identity(inputs) si las dimensiones coinciden
        #   - Conv 1×1 con stride adaptado si las dimensiones cambian
        
        # Suma ambas ramas y aplica ReLU al resultado
        return F.relu(self.main_layers(inputs) + self.skip_connection(inputs))
