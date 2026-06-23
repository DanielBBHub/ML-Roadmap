# Resumen de apuntes: Fundamentos de PyTorch


## 1) `Pytorch_Tensors.py`

### Idea principal
Introducción práctica a los **tensores en PyTorch** como estructura de datos base (análoga a arrays multidimensionales de NumPy).

### Puntos clave
- Creación de tensores con `torch.tensor(...)`.
- Inspección de:
  - forma (`.shape`)
  - tipo de dato (`.dtype`)
- Indexación y slicing igual que en NumPy.
- Operaciones element-wise y algebra lineal:
  - suma/escalado
  - exponencial (`.exp()`)
  - multiplicación matricial (`@`) y traspuesta (`.T`)
- Conversión entre PyTorch y NumPy:
  - tensor → numpy con `.numpy()`
  - numpy → tensor con `torch.tensor(...)`
- Nota de precisión:
  - Se recomienda `float32` para reducir uso de memoria y mantener rendimiento adecuado en entrenamiento.
- Operaciones in-place (modifican el tensor original), por ejemplo:
  - asignación por índice
  - `relu_()` para truncar negativos a 0.

---

## 2) `Pytorch_HW_ACC.py`

### Idea principal
Uso de **aceleración por hardware** (GPU/Apple Silicon) y selección dinámica de dispositivo para ejecutar operaciones más rápido.

### Puntos clave
- Verificación de soporte de CUDA (`torch.cuda.is_available()`) y alternativa MPS (`torch.backends.mps.is_available()`).
- Selección de dispositivo en tiempo de ejecución:
  - `cuda` si hay GPU NVIDIA compatible
  - `mps` si hay soporte Metal (Apple)
  - `cpu` como fallback
- Movimiento de tensores al dispositivo con:
  - `.to(device)`
  - creación directa con `device=device`
- Verificación del dispositivo donde reside un tensor mediante `.device`.
- Ejemplo de multiplicación matricial en el dispositivo seleccionado.
- Benchmark simple CPU vs GPU con `time.perf_counter()` y (en CUDA) `torch.cuda.synchronize()` para medir tiempos reales.
- Conclusión: en cargas suficientemente grandes, la GPU suele ofrecer mejor tiempo que CPU.

### Advertencia útil
En el ejemplo hay una línea que crea tensor explícitamente en `device="cuda"`; en un entorno sin CUDA eso fallaría. Para mayor portabilidad, conviene usar siempre `device=device`.

---

## 3) `Pytorch_Autograd.py`

### Idea principal
Introducción a **autograd** en PyTorch para cálculo automático de gradientes y pasos básicos de descenso de gradiente.

### Puntos clave
- Activar gradiente con `requires_grad=True`.
- Construcción del grafo computacional al definir operaciones (ej.: `f = x**2`).
- Cálculo de gradiente con `f.backward()` y lectura en `x.grad`.
- Paso de optimización manual:
  - `x -= learning_rate * x.grad` dentro de `torch.no_grad()`.
- Diferencia entre:
  - `torch.no_grad()`: desactiva tracking temporalmente (útil en inferencia/updates).
  - `.detach()`: crea vista/copia desligada del grafo para operar sin propagar gradiente.
- Importancia de resetear gradientes con `x.grad.zero_()` para evitar acumulación entre iteraciones.
- Esquema de bucle de entrenamiento básico:
  1. forward
  2. backward
  3. update
  4. zero grad
- Nota sobre operaciones in-place y autograd:
  - Dependiendo de si la operación guarda entradas/salidas para backward, modificar en sitio puede lanzar `RuntimeError`.

---

## Relación entre los tres archivos
- `Pytorch_Tensors.py` cubre la **base de manipulación de datos**.
- `Pytorch_HW_ACC.py` muestra **dónde ejecutar** esos cálculos para acelerar entrenamiento/inferencia.
- `Pytorch_Autograd.py` explica **cómo aprender parámetros** mediante gradientes.

En conjunto, forman una introducción coherente a los pilares iniciales de PyTorch: **tensores, dispositivo y autograd**.
