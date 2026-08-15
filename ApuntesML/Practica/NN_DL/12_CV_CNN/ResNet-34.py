import torch
from torch import torch.nn as nn
from Res_unit import ResidualUnit

class ResNet34(nn.Module):
    """
    Implementa ResNet-34: una red neuronal convolucional profunda con 34 capas.
    
    ¿Qué es ResNet-34?
    ==================
    ResNet (Residual Network) es una arquitectura revolucionaria que permite
    entrenar redes neuronales mucho más profundas que antes gracias a las
    SKIP CONNECTIONS (conexiones residuales).
    
    El "34" proviene de contar las capas:
    - 1 capa convolucional inicial
    - 33 capas en bloques residuales (16 bloques × 2 capas conv + ajustes dimensionales)
    - 1 capa fully connected de clasificación
    = 34 capas totales
    
    Arquitectura de ResNet-34
    ==========================
    La red está dividida en 4 etapas principales:
    
    Etapa 1 (Entrada):
    - Conv 7×7, stride=2 → reduce de 224×224 a 112×112
    - MaxPool 3×3, stride=2 → reduce de 112��112 a 56×56
    - Resultado: 56×56×64 mapas de características
    
    Etapa 2 (3 bloques residuales):
    - 3 bloques residuales con 64 canales
    - Mantiene tamaño: 56×56×64
    
    Etapa 3 (4 bloques residuales):
    - 4 bloques residuales con 128 canales
    - Primer bloque reduce tamaño: 56×56 → 28×28
    - Mantiene: 28×28×128
    
    Etapa 4 (6 bloques residuales):
    - 6 bloques residuales con 256 canales
    - Primer bloque reduce tamaño: 28×28 → 14×14
    - Mantiene: 14×14×256
    
    Etapa 5 (3 bloques residuales):
    - 3 bloques residuales con 512 canales
    - Primer bloque reduce tamaño: 14×14 → 7×7
    - Mantiene: 7×7×512
    
    Clasificación:
    - AdaptiveAvgPool: 7×7×512 → 1×1×512
    - Fully Connected: 512 → 10 (número de clases)
    
    Diagrama visual:
    ================
    Entrada (224×224×3)
        ↓
    [Conv 7×7, stride=2] → [BN] → [ReLU] → [MaxPool stride=2]
        ↓
    56×56×64 (3 bloques ResNet, stride=1)
        ↓
    28×28×128 (4 bloques ResNet, primer stride=2, resto stride=1)
        ↓
    14×14×256 (6 bloques ResNet, primer stride=2, resto stride=1)
        ↓
    7×7×512 (3 bloques ResNet, primer stride=2, resto stride=1)
        ↓
    [AdaptiveAvgPool] → [Flatten] → [Linear(512, 10)]
        ↓
    Predicción (10 clases)
    
    Ventajas de ResNet-34
    =====================
    1. Skip Connections: Permite entrenar redes profundas sin vanishing gradients
    2. Batch Normalization: Estabiliza el entrenamiento
    3. Residual Learning: Aprende diferencias en lugar de características nuevas
    4. Arquitectura escalable: Mismo patrón se usa en ResNet-50, ResNet-101, etc.
    """
    
    def __init__(self):
        super().__init__()
        
        # ================================================================
        # ETAPA 1: Convolución inicial y max pooling
        # ================================================================
        # Esta etapa prepara la imagen para la red profunda
        # Reduce el tamaño de 224×224 a 56×56 (1/4 del tamaño original)
        layers = [
            # Convolución inicial: extrae características iniciales
            nn.Conv2d(
                in_channels=3,        # Entrada: imágenes RGB (3 canales de color)
                out_channels=64,      # Salida: 64 mapas de características
                kernel_size=7,        # Kernel grande (7×7) para detectar patrones grandes
                stride=2,             # Stride 2: reduce el tamaño a la mitad
                                      # 224×224 → 112×112
                padding=3,            # Padding para mantener proporciones
                                      # padding = (kernel_size - 1) / 2 = (7-1)/2 = 3
                bias=False),          # Sin bias porque BatchNorm lo hace innecesario
            
            # Batch Normalization: normaliza las salidas de la convolución
            # Acelera el entrenamiento y permite usar tasas de aprendizaje más altas
            nn.BatchNorm2d(num_features=64),
            
            # ReLU: función de activación
            # Introduce no-linealidad permitiendo a la red aprender relaciones complejas
            nn.ReLU(),
            
            # Max Pooling: reduce el tamaño espacialmente
            # Toma el valor máximo en cada ventana 3×3
            nn.MaxPool2d(
                kernel_size=3,       # Ventana de 3×3
                stride=2,            # Stride 2: reduce tamaño a la mitad
                                     # 112×112 → 56×56
                padding=1),          # Padding para evitar pérdida de información en bordes
        ]
        
        prev_filters = 64  # Número de canales después de la etapa inicial
        
        # ================================================================
        # ETAPA 2: Bloques residuales
        # ================================================================
        # Aquí está el corazón de ResNet: los bloques residuales
        
        # La arquitectura de ResNet-34 está definida por esta secuencia:
        # [64]*3:   3 bloques residuales con 64 canales (mantiene tamaño 56×56)
        # [128]*4:  4 bloques residuales con 128 canales (reduce a 28×28, luego mantiene)
        # [256]*6:  6 bloques residuales con 256 canales (reduce a 14×14, luego mantiene)
        # [512]*3:  3 bloques residuales con 512 canales (reduce a 7×7, luego mantiene)
        #
        # Total de bloques: 3+4+6+3 = 16 bloques residuales
        # Cada bloque tiene 2 capas convolucionales
        # Total de capas conv: 16 × 2 = 32 capas conv + 1 inicial + 1 FC = 34 capas
        
        for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
            # Lógica del stride:
            # - Si el número de filtros es IGUAL al anterior: stride=1 (mantiene tamaño)
            # - Si el número de filtros CAMBIA: stride=2 (reduce tamaño a la mitad)
            #
            # Esto es importante porque:
            # 1. Cuando cambio de 64→128 canales, necesito reducir tamaño
            #    (mantener proporciones CPU/memoria)
            # 2. Cuando mantengo 128→128 canales, no cambio tamaño
            #    (extraigo más características del mismo nivel)
            
            stride = 1 if filters == prev_filters else 2
            
            # Crea un bloque residual con los parámetros calculados
            # El bloque se encarga de adaptar dimensiones si es necesario (via skip connection)
            layers.append(ResidualUnit(prev_filters, filters, stride=stride))
            
            # Actualiza el número de filtros para el siguiente bloque
            prev_filters = filters
        
        # ================================================================
        # ETAPA 3: Pooling global y clasificación
        # ================================================================
        # Reduce los mapas de características a un vector único
        # Luego lo usa para clasificar
        
        layers += [
            # Adaptive Average Pooling: promedia cada mapa de características
            # Reduce dimensiones espaciales a 1×1
            # Entrada: (batch, 512, 7, 7)
            # Salida:  (batch, 512, 1, 1)
            #
            # "Adaptive" significa que ajusta automáticamente el kernel
            # para producir exactamente el tamaño de salida deseado
            nn.AdaptiveAvgPool2d(output_size=1),
            
            # Flatten: convierte (batch, 512, 1, 1) en (batch, 512)
            # Necesario para alimentar la capa fully connected
            nn.Flatten(),
            
            # Capa de clasificación final: fully connected
            # Transforma 512 características en 10 predicciones de clase
            # (ej: para CIFAR-10 con 10 clases)
            #
            # LazyLinear infiere automáticamente el tamaño de entrada (512)
            # No necesita que especifiques in_features
            nn.LazyLinear(10),
        ]
        
        # Combina todas las capas en una secuencia
        # Sequential aplica cada capa secuencialmente en el forward pass
        self.resnet = nn.Sequential(*layers)
    
    def forward(self, inputs):
        """
        Forward pass: procesa la entrada a través de toda la red ResNet-34.
        
        Args:
            inputs: Tensor de entrada con forma (batch_size, 3, 224, 224)
                   Imágenes RGB de 224×224 píxeles
        
        Returns:
            output: Tensor de predicción con forma (batch_size, 10)
                   10 probabilidades (una por clase)
        """
        return self.resnet(inputs)
