# Resumen detallado — ApuntesML/Practica/NN_DL/11_Training_DNN

Este archivo es un resumen más extenso y estructurado del directorio `ApuntesML/Practica/NN_DL/11_Training_DNN`. Incluye explicación de los conceptos principales, qué contiene cada fichero con detalles prácticos y una guía rápida para ejecutar los ejemplos y aplicar las técnicas en tus propios entrenamientos.

---

## 1. Visión general
El directorio combina:
- Un documento teórico extenso (`Training Deep Neural Nets.md`) que cubre problemas y soluciones al entrenar DNNs (inicialización, activaciones, normalización, optimizadores, scheduling, regularización, transfer learning, etc.).
- Scripts en PyTorch con ejemplos prácticos (inicialización, activaciones, batch-norm, layer-norm, recorte de gradientes, optimizadores, regularización).
- Utilidades y ejercicios (notebook y scripts de ejercicio) para practicar y reproducir entrenamientos.

El material está pensado para pasar de la teoría a la práctica con ejemplos reproducibles y utilidades de apoyo (guardado de pesos, funciones de entrenamiento).

---

## 2. Fichero principal (teoría)
- **Training Deep Neural Nets.md**  
  Contenido:
  - Problema de gradientes que explotan/desaparecen: causas y consecuencias.
  - Inicializaciones: Glorot (Xavier) y He (Kaiming) — cuándo usar cada una, fórmulas y recomendaciones.
  - Funciones de activación: ReLU y variantes (LeakyReLU, ELU, SELU), GELU, Swish, SwiGLU, Mish — ventajas, desventajas y recomendaciones de uso.
  - Normalización: BatchNorm (fórmulas, parámetros γ/β y buffers running), LayerNorm — diferencias y cuándo optar por cada una.
  - Técnicas de estabilización: gradient clipping, max-norm.
  - Transfer learning y preentrenamiento: estrategias para reutilizar pesos y recomendaciones (congelar, reajustar LR, preprocesado de entrada).
  - Optimizadores: SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam y variantes (AdaMax, NAdam, AdamW) — ecuaciones y comportamiento.
  - Schedulers: decay exponencial, cosine annealing (y reinicios), ReduceLROnPlateau, warmup, 1cycle.
  - Regularización: L1/L2, dropout, MC-dropout, alpha-dropout para SELU, max-norm.
  - Guía práctica/tabla de hiperparámetros recomendados (inicializador, activación, normalización, regularización, optimizador, scheduling).

---

## 3. Scripts y utilidades (descripciones detalladas)
A continuación se listan los ficheros con sus responsabilidades y puntos importantes a revisar.

- **Activation.py**  
  - Ejemplos de uso de `nn.LeakyReLU` y `nn.ELU` en modelos secuenciales.
  - Muestra aplicación de inicialización Kaiming (`nn.init.kaiming_uniform_`) adecuada cuando se usan activaciones tipo ReLU/LeakyReLU.
  - Recomendación: revisar el parámetro `negative_slope` para LeakyReLU y usar `nonlinearity="leaky_relu"` en la inicialización.

- **DDN_Init.py**  
  - Muestra distintas maneras de inicializar capas (`kaiming_uniform_`, zeros, orthogonal).
  - Incluye función `use_he_init(module)` y uso de `model.apply(use_he_init)` para aplicar la inicialización a todas las capas relevantes.
  - Observación: inicializar biases a cero es estándar; cuidado con inicializaciones que rompan la simetría.

- **Batch_Norm.py**  
  - Ejemplo completo de pipeline: DataLoaders (FashionMNIST), modelo con `nn.BatchNorm1d`, training helper (`ModelUtl.Train.train_minibatch_gd`), uso de `torchmetrics.Accuracy` y guardado de pesos vía `SNL.saveWeights`.
  - Muestra cómo inspeccionar parámetros (`named_parameters`) y buffers (`named_buffers`) de BatchNorm (γ, β, running_mean, running_var, num_batches_tracked).
  - Útil como punto de partida reproducible para entrenamientos con BN.

- **Faster_optims.py**  
  - Declaraciones de optimizadores PyTorch (SGD con momentum, nesterov, Adagrad, RMSprop, Adam, NAdam, Adamax, AdamW).
  - Recomendación: ajustar learning rate y weight decay según optimizador (p. ej. AdamW combina bien con weight decay separado).

- **Grad_Clip.py**  
  - Ejemplo práctico de `nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` dentro del bucle de entrenamiento.
  - Útil para RNNs o redes profundas donde la explosión de gradientes ocurre.

- **Layer_Norm.py**  
  - Uso de `nn.LayerNorm` y la especificación de las dimensiones a normalizar (importante en inputs por ejemplo de forma [B, C, H, W]).
  - Indicado para arquitecturas donde BN no es apropiado (transformers, secuencias, batch pequeño).

- **Reg.py**  
  - Implementaciones prácticas de regularización:
    - L2: manual y usando parameter groups en el optimizador (para evitar aplicar decay a biases o a BN).
    - L1: implementación manual (sum param.abs()).
    - Dropout: ejemplos de `nn.Dropout` en arquitectura secuencial.
    - MC Dropout: cómo forzar dropout activo en evaluación para estimar incertidumbre (repetir inputs y promediar softmax).
    - Max-norm: función `apply_max_norm(model, max_norm=2, ...)` que reescala pesos.
  - Muy útil para entender cómo aplicar regularización "a la carta" en PyTorch.

- **Trans_Learning.py**  
  - Contiene utilidades/ejemplos para transfer learning y reutilización de capas preentrenadas.  
  - Recomendaciones típicas: congelar capas iniciales, ajustar LR para capas nuevas, adaptar tamaño de entrada.

- **SNL.py**  
  - Funciones utilitarias (ej. `saveWeights`) — revisa este fichero si vas a guardar/recuperar pesos.

- **ModelUtl/**  
  - Directorio con helpers de entrenamiento (train loops, evaluación). `Batch_Norm.py` importa `ModelUtl.Train.train_minibatch_gd`.

- **Ejercicio 8.ipynb** y **Ex.py**  
  - Notebook y script de ejercicios para practicar (CIFAR10: pruebas con Swish/SiLU, He init, NAdam, early stopping, comparar con BN/SELU, MC dropout, 1cycle).
  - `Ex.py` contiene respuestas a preguntas y una práctica paso a paso.

---

## 4. Recomendaciones prácticas y "receta" rápida
- Inicialización: He/Kaiming para ReLU/SiLU; Xavier para activaciones simétricas (tanh). Aplicar con `model.apply(use_he_init)`.
- Activación: ReLU por defecto; SiLU (Swish) o GELU para modelos grandes; LeakyReLU si aparecen neuronas muertas; SELU con LeCun init y alpha-dropout para auto-normalización (MLP puro).
- Normalización: BatchNorm para CNNs y batch > 1; LayerNorm para transformers y batch pequeño.
- Optimizer: AdamW o Nesterov-SGD; probar AdamW + scheduler 1cycle.
- Scheduler: combinar warmup (unos epochs) + 1cycle o cosine annealing con reinicios si conviene explorar varios mínimos.
- Regularización: weight decay (L2) en parámetros seleccionados, dropout (o alpha-dropout con SELU), early stopping + monitor en validación.
- Grad clip: usar `clip_grad_norm_` si gradientes explotan (umbral típico 1.0–5.0).

Tabla de partida recomendada
| Hiperparámetro | Valor inicial sugerido |
|---|---|
| Inicializador | He/Kaiming |
| Activación | ReLU (SiLU para redes profundas) |
| Normalización | BatchNorm (LayerNorm si corresponde) |
| Regularización | weight_decay=1e-4, dropout p=0.2 |
| Optimizador | AdamW (lr inicial 1e-3) |
| Scheduler | 1cycle o ReduceLROnPlateau |
| Grad clip | clip_grad_norm_(..., max_norm=1.0) |

---

## 5. Cómo ejecutar el ejemplo de Batch_Norm.py (rápido)
1. Crear y activar entorno virtual, instalar dependencias:
```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision torchmetrics
```
2. Ejecutar:
```bash
python ApuntesML/Practica/NN_DL/11_Training_DNN/Batch_Norm.py
```
- Revisa `ModelUtl` y `SNL.saveWeights` si necesitas adaptar rutas de guardado.

---

Si quieres, escribo directamente este resumen como un fichero en tu repo (creando/actualizando `ApuntesML/Practica/NN_DL/11_Training_DNN/README_SUMMARY.md`) o extraigo y muestro fragmentos relevantes (por ejemplo: la función `apply_max_norm` completa, el loop de MC Dropout o el contenido del notebook `
