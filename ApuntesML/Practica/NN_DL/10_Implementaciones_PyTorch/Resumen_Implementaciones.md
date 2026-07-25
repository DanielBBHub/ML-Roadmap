# Resumen de apuntes: Implementaciones (PyTorch)

Este documento resume el contenido y la evolución del directorio `ApuntesML/Practica/NN_DL/10_Implementaciones_PyTorch`.

## 1) `ModelUtl/Train.py`

### Idea principal
Centraliza utilidades de entrenamiento y preprocesado para distintos tipos de modelos (entrada simple y multi-input).

### Puntos clave
- Incluye la función de particionado y normalización de datos (`train_test_val`).
- Implementa bucles de entrenamiento mini-batch para:
  - modelos estándar de una entrada,
  - modelos con múltiples entradas (por ejemplo `X_wide` + `X_deep`).
- Gestiona ciclo típico de entrenamiento:
  - `model.train()` en fase de entrenamiento,
  - cálculo de pérdida y backpropagation,
  - paso del optimizador,
  - validación por época con `model.eval()`.
- Añade soporte para evaluación durante el entrenamiento y lógica de early stopping basada en validación.

---

## 2) `ModelUtl/Eval.py`

### Idea principal
Agrupa funciones de evaluación para mantener el código de entrenamiento limpio y reutilizable.

### Puntos clave
- Evaluación manual de métricas sobre `DataLoader`.
- Evaluación con `torchmetrics` mediante una rutina dedicada (`evaluate_tm`).
- Uso de `model.eval()` y `torch.no_grad()` para evitar cálculo de gradientes durante evaluación.
- Evolución reciente orientada a soportar entradas más complejas (tensores o diccionarios en escenarios multi-input).

---

## 3) `LinearRegTensor.py`

### Idea principal
Implementación de regresión lineal **desde cero** con tensores y `autograd`.

### Puntos clave
- Parámetros manuales entrenables: pesos `w` y sesgo `b`.
- Entrenamiento explícito:
  1. forward,
  2. cálculo de loss MSE,
  3. backward,
  4. actualización manual,
  5. reseteo de gradientes.
- Incluye inferencia final sobre muestras de test.

### Valor didáctico
Permite entender qué abstrae PyTorch cuando se pasa a `nn.Module` y optimizadores de alto nivel.

---

## 4) `LinearRegAPI.py`

### Idea principal
Regresión lineal con API de alto nivel (`nn.Linear`, `SGD`, `MSELoss`).

### Puntos clave
- Define el modelo como módulo de PyTorch.
- Muestra inspección de parámetros del modelo.
- Usa función de entrenamiento tipo Batch Gradient Descent (`train_bgd`).
- Incluye ejemplo de inferencia tras entrenamiento.

---

## 5) `RegMLP.py`

### Idea principal
Modelo MLP para regresión con `nn.Sequential`.

### Arquitectura típica
- `Linear(n_features, 50)` + `ReLU`
- `Linear(50, 40)` + `ReLU`
- `Linear(40, 1)`

### Puntos clave
- Reutiliza utilidades de preprocesado y entrenamiento.
- Cambia de modelo lineal a no lineal para capturar relaciones más complejas.

---

## 6) `MiniBatchGradient.py`

### Idea principal
Entrenamiento mini-batch con `TensorDataset` + `DataLoader`, con foco en rendimiento y aceleración.

### Puntos clave
- Selección dinámica de dispositivo: `cuda` / `mps` / `cpu`.
- Configuración de `DataLoader` con opciones de rendimiento:
  - `pin_memory`,
  - `num_workers`,
  - `prefetch_factor`,
  - `persistent_workers`.
- Entrenamiento por lotes con `Adam` y `MSELoss`.
- Buen ejemplo de transición hacia pipeline más realista.

---

## 7) `WideNDeep.py`

### Idea principal
Implementa variantes de arquitecturas **Wide & Deep** y dataset personalizado para multi-input.

### Modelos incluidos
- **`WideAndDeep`**: usa toda la entrada para la rama deep y concatena con la entrada original antes de la salida.
- **`WideAndDeepV2`**: separa características en rama wide y deep usando slicing (`X[:, :5]` y `X[:, 2:]`).
- **`WideAndDeepV3`**: versión de múltiples entradas explícitas en `forward(X_wide, X_deep)`.

### Dataset personalizado
- **`WideAndDeepDataset`** devuelve:
  - `{"X_wide": ..., "X_deep": ...}`
  - etiqueta `y`
- Esto reduce errores de orden de argumentos y facilita entrenar/evaluar modelos multi-input.

### Aprendizaje práctico relevante
- Al separar features en wide/deep hay que vigilar consistencia de dimensiones y posibles solapamientos de columnas.

---

## 8) `NonSecuential.py`

### Idea principal
Script de experimentación y comparación entre arquitecturas no secuenciales.

### Qué hace
- Carga y prepara California Housing.
- Construye datasets/loaders estándar y loaders para wide-deep (incluyendo dataset con diccionario de entradas).
- Entrena tres modelos en paralelo conceptual:
  - `WideAndDeep`,
  - `WideAndDeepV2`,
  - `WideAndDeepV3`.
- Usa **un optimizador por modelo** (buena práctica clave).
- Realiza inferencia comparativa final sobre muestras de test.

### Valor didáctico
Es el archivo donde se integran los conceptos nuevos del directorio: multi-input, comparación de arquitecturas y evaluación práctica.

---

## Panorama general actual del directorio

La progresión ya no es solo “lineal → MLP → mini-batch”, sino:

1. **Fundamentos de entrenamiento** (`LinearRegTensor`, `LinearRegAPI`).
2. **Generalización a redes profundas** (`RegMLP`, `MiniBatchGradient`).
3. **Modularización de utilidades** (`ModelUtl/Train.py`, `ModelUtl/Eval.py`).
4. **Arquitecturas no secuenciales y multi-input** (`WideNDeep.py`).
5. **Comparación experimental integrada** (`NonSecuential.py`).

En conjunto, el directorio refleja una evolución clara desde implementaciones básicas de regresión hasta pipelines de entrenamiento más completos y flexibles para modelos
