# Resumen de apuntes: Implementaciones (NN_DL)


## 1) `train_test_val.py`

### Idea principal
Función auxiliar para preparar datos de entrenamiento, validación y test, además de normalizar variables de entrada.

### Puntos clave
- Divide el dataset en:
  - **train: 75%**
  - **validation: 15%**
  - **test: 10%**
- Convierte las particiones de entrada a `torch.FloatTensor`.
- Calcula **media** y **desviación estándar** usando solo `x_train`.
- Normaliza `x_train`, `x_val` y `x_test` con esas estadísticas de entrenamiento (evita fuga de información).
- Convierte etiquetas a tensores columna con `.reshape(-1, 1)` para que coincidan con la salida de modelos de regresión.

---

## 2) `LinearRegTensor.py`

### Idea principal
Implementación de una regresión lineal **desde cero** usando tensores y `autograd`.

### Puntos clave
- Define manualmente parámetros entrenables:
  - pesos `w`
  - sesgo `b`
  con `requires_grad=True`.
- Entrena con ciclo explícito de gradiente:
  1. forward (`y_pred = x_train @ w + b`)
  2. loss MSE manual
  3. backward (`loss.backward()`)
  4. actualización de parámetros dentro de `torch.no_grad()`
  5. `zero_()` de gradientes
- Incluye lógica sencilla de **early stopping** basada en cambio mínimo de pérdida.
- Realiza inferencia final sobre muestras de test.

### Nota
El script usa `train_test_val(...)`, por lo que requiere importar esa función correctamente para ejecución directa.

---

## 3) `LinearRegAPI.py`

### Idea principal
Regresión lineal usando la **API de alto nivel** de PyTorch (`nn.Module`, optimizadores y función de pérdida).

### Puntos clave
- Modelo definido con `nn.Linear(in_features=n_features, out_features=1)`.
- Inspección de parámetros del modelo:
  - `model.bias`
  - `model.weight`
  - iteración sobre `model.parameters()`.
- Configuración de entrenamiento:
  - optimizador `torch.optim.SGD`
  - pérdida `nn.MSELoss()`.
- Función `train_bgd(...)` que implementa **Batch Gradient Descent**:
  - forward
  - cálculo de loss
  - backward
  - `optimizer.step()`
  - `optimizer.zero_grad()`.
- Inferencia final sobre nuevas muestras.

### Enfoque didáctico
Muestra la transición desde cálculo manual con tensores a un enfoque más estándar y mantenible con API de PyTorch.

---

## 4) `RegMLP.py`

### Idea principal
Modelo de regresión con red neuronal **MLP** usando `nn.Sequential`.

### Arquitectura
- Entrada: `n_features`
- `Linear(n_features, 50)` + `ReLU`
- `Linear(50, 40)` + `ReLU`
- `Linear(40, 1)`

### Puntos clave
- Reutiliza `train_test_val(...)` para preparar datos.
- Reutiliza `train_bgd(...)` (importado desde `LinearRegAPI`) para entrenar el MLP.
- Usa `SGD` y `MSELoss`.
- Ejemplo de inferencia en modo `torch.no_grad()`.

### Observación
Aunque se llama a una rutina de entrenamiento “bgd”, funciona para cualquier `nn.Module` siempre que reciba entradas/salidas compatibles.

---

## 5) `MiniBatchGradient.py`

### Idea principal
Entrenamiento de red con **mini-batches** usando `TensorDataset` + `DataLoader`, con soporte de CPU/GPU/MPS y optimizaciones de carga.

### Puntos clave
- Selección dinámica de dispositivo:
  - `cuda` / `mps` / `cpu`.
- Construcción de `DataLoader` con parámetros de rendimiento:
  - `batch_size=32`
  - `shuffle=True`
  - `pin_memory` (especialmente útil en CUDA)
  - `num_workers`
  - `prefetch_factor`
  - `persistent_workers`
- Loop de entrenamiento por lotes en `train_minibatch_gd(...)`:
  - mueve batch a dispositivo (`non_blocking=True`)
  - calcula predicción/loss
  - backward + step + zero_grad
  - reporta pérdida media por época.
- Modelo MLP similar al de `RegMLP.py`.
- Cambia optimizador a `Adam` con `lr` más pequeño (`0.001`) para estabilidad.
- Incluye `mp.freeze_support()` para evitar problemas de multiprocessing en algunos entornos.

### Comentario práctico
Este archivo muestra un enfoque más cercano a entrenamiento real en producción que las versiones por batch completo.

---

## Relación entre los archivos
- `train_test_val.py` aporta el **preprocesado y particionado** común.
- `LinearRegTensor.py` enseña el enfoque **manual** (nivel bajo).
- `LinearRegAPI.py` migra a enfoque **API PyTorch** (nivel alto).
- `RegMLP.py` amplía de regresión lineal a una **red multicapa**.
- `MiniBatchGradient.py` introduce **mini-batch training** y optimización de pipeline de datos para hardware acelerado.

En conjunto, el directorio muestra una progresión clara: **preparación de datos → modelo lineal manual → API de PyTorch → MLP → entrenamiento eficiente con DataLoader**.
