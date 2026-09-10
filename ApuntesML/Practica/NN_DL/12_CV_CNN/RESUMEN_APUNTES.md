# 📚 Apuntes — Tema 12: Visión por Computador con CNNs

> Documento de consulta rápida. Objetivo: de un vistazo saber **qué archivo abrir** según la duda que tengas. Cada sección resume el archivo correspondiente; no sustituye al código/teoría original, solo indexa y destaca lo importante.

---

## 🔍 Índice rápido — "tengo una duda sobre..."

| Duda / tema | Archivo a abrir |
|---|---|
| Teoría de capas convolucionales, fórmula matemática de la convolución | `Deep computer vision....md` (§ Capas convolucionales) |
| Teoría de pooling (max/avg), invarianza | `Deep computer vision....md` (§ Pooling) + `Pooling.py` (código) |
| Comparativa de arquitecturas famosas (LeNet, AlexNet, GoogLeNet, ResNet, Xception, SENet, MobileNet, EfficientNet, ConvNeXt) | `Deep computer vision....md` (§ Arquitecturas) |
| RAM/GPU necesaria para entrenar/inferir con una CNN | `Deep computer vision....md` (§ Requerimientos de RAM) + `Ejercicios/ex.md` (ejercicio 2) |
| Cómo crear una capa `Conv2d` en PyTorch, qué es padding "valid"/"same", forma de pesos/bias | `PyTorch_ConvLayer.py` |
| Cómo implementar una convolución **separable** (depthwise + pointwise, como en Xception/MobileNet) | `Conv_separables.py` |
| Cómo usar `MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d` (global avg pool) y cómo implementar **depth pooling** manual | `Pooling.py` |
| Ejemplo completo de una CNN sencilla (tipo LeNet) para FashionMNIST | `Sample_CNN.py` |
| Qué es una **skip connection** / residual unit y cómo se implementa en PyTorch | `Res_unit.py` |
| Cómo se ensambla una arquitectura completa tipo **ResNet-34** a partir de bloques residuales | `ResNet-34.py` |
| Cómo cargar y usar un modelo **preentrenado** de TorchVision para inferencia directa (sin reentrenar) | `Pretrained.py` |
| Cómo hacer **transfer learning / fine-tuning** real (congelar capas, cambiar la cabeza, data augmentation, entrenar) | `Transf_pretrained.py` |
| Lista de las 102 clases del dataset Flowers102 | `flowerclasses.py` |
| Cómo combinar **clasificación + localización** (bounding box) en un mismo modelo (multi-output) | `Class_Local.py` + `Deep computer vision....md` (§ Clasificación y Localización, IoU/GIoU/CIoU) |
| Teoría de detección de objetos (non-max suppression, FCN, YOLO), seguimiento (DeepSORT) y segmentación semántica | `Deep computer vision....md` (§ Detección de objetos / Seguimiento / Segmentación) |
| Respuestas escritas a los ejercicios teóricos 1-7 del capítulo | `Ejercicios/ex.md` |
| Ejercicio 8: CNN propia para MNIST (comparando salida FCN vs. capa densa) | `Ejercicios/Ejercicios-8-9-10.ipynb` (celdas 4-13) |
| Ejercicio 9: transfer learning a dataset propio | Aplicado en el repo [ReconocimientoComponentes](https://github.com/DanielBBHub/ReconocimientoComponentes) (celdas 14-22 del notebook quedan vacías, ver nota) |
| Ejercicio 10: tutorial oficial de fine-tuning para **detección de objetos** (Faster R-CNN / Mask R-CNN sobre PennFudan) | `Ejercicios/Ejercicios-8-9-10.ipynb` (celdas 23-40) |
| Utilidades boilerplate de torchvision para entrenar/evaluar modelos de detección (COCO metrics, collate_fn, etc.) | `Ejercicios/{engine,utils,coco_utils,coco_eval,transforms}.py` |
| Diagramas de arquitecturas (Inception, GoogLeNet, ResNet, tabla de modelos por tamaño) | `img/*.png` |

---

## 1. Teoría general

### `Deep computer vision using convolutional neural networks.md`
Apunte teórico principal del tema (272 líneas). Cubre, en orden:

- **Capas convolucionales**: campo receptivo local y jerárquico (bajo nivel → alto nivel), fórmula de conexión entre capas con stride $(s_h, s_w)$ y padding "same" para conservar tamaño. Filtros/kernels como pesos compartidos; ejemplo de filtro vertical/horizontal.
- **Mapas de características**: una capa convolucional puede tener varios filtros → varios mapas; todas las neuronas de un mismo mapa comparten parámetros. Fórmula completa de $z_{i,j,k}$ (ecuación de salida de una neurona convolucional).
- **Pooling**: reduce coste computacional/memoria/parámetros; sin pesos ni bias; aporta invarianza a pequeñas traslaciones pero es destructivo (pierde info espacial).
- **Arquitecturas clásicas** con tablas capa-por-capa: **LeNet-5**, **AlexNet** (dropout + data augmentation + LRN), **GoogLeNet** (inception modules, cuellos de botella 1×1), **ResNet** (skip connections, aprendizaje residual $h(x)-x$), **Xception** (convoluciones separables: filtro espacial + filtro 1×1 entre canales), **SENet** (bloque SE: global avg pool → capa cuello de botella ReLU → sigmoide, recalibra mapas de características), **MobileNet/SqueezeNet/ShuffleNet/MNasNet** (ligeras), **EfficientNet** (compound scaling: $\phi$ escala profundidad/anchura/resolución con $\alpha,\beta,\gamma$), **ConvNeXt** (ResNet + ideas de transformers de visión).
- **RAM GPU**: ejemplo numérico de cuánta memoria consume una capa convolucional (parámetros vs. activaciones); en inferencia solo hacen falta ~2 capas en memoria simultáneamente, en entrenamiento hace falta guardar todo el forward pass para el backward. Trucos si se agota memoria: reducir batch, aumentar stride, quitar capas, quantizar, repartir entre CPU/GPU.
- **Clasificación y Localización**: se añade una cabeza extra que predice 4 números (bounding box). Métrica de evaluación: **IoU** (falla si no hay solape → gradiente 0), **GIoU** (soluciona con caja envolvente $S$), **CIoU** (añade distancia de centros + ratio de aspecto). Uso de `torchvision.tv_tensors.BoundingBoxes`.
- **Detección de objetos**: enfoque clásico de ventana deslizante + "objectness score" + **non-max suppression**; problema de coste en inferencia. Solución: **redes totalmente convolucionales (FCN)** — sustituir capas densas por convolucionales equivalentes, permite procesar imágenes de cualquier tamaño. **YOLO**: predicción por celda, coordenadas relativas a la celda, 2 cajas por celda, distribución de clases por celda (no por caja).
- **Seguimiento de objetos**: **DeepSORT** (filtro de Kalman + modelo de similitud + algoritmo húngaro), y **Bot-SORT** (mejora de Ultralytics con compensación de cámara).
- **Segmentación semántica**: cada píxel → clase; problema de resolución espacial perdida; solución con capas de **convolución traspuesta** (upsampling) + skip connections para recuperar precisión.

> Contiene también 3 imágenes de referencia (`img/inception_module.png`, `img/GoogLeNet.png`, `img/ResNet.png`, `img/lista_modelos.png`) e incluye tablas markdown con las arquitecturas de LeNet-5 y AlexNet capa por capa.

---

## 2. Código: fundamentos de capas (PyTorch)

### `PyTorch_ConvLayer.py`
Primer contacto práctico con `nn.Conv2d`.
- Carga imágenes de ejemplo de sklearn, las normaliza (0-1) y permuta a formato `[batch, canales, alto, ancho]` (el que espera PyTorch).
- `T.CenterCrop` para recortar.
- Crea `nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7)` → por defecto `padding=0` ("valid"): la imagen de salida **encoge** (pierde píxeles en los bordes).
- Con `padding="same"` la salida conserva el tamaño de entrada (rellena con ceros).
- Inspecciona la forma de los parámetros: pesos `[out_channels, in_channels, kh, kw]`, bias `[out_channels]`.
- Nota: hace falta activación después de cada capa conv (igual que en redes densas) para poder aprender patrones no lineales.

### `Conv_separables.py`
Implementación de **convolución separable en profundidad** (depthwise separable, base de Xception/MobileNet) como módulo custom `SeparableConv2d`:
- `depthwise_conv`: `nn.Conv2d(in_channels, in_channels, kernel_size, groups=in_channels)` → un filtro espacial por canal, sin mezclar canales.
- `pointwise_conv`: `nn.Conv2d(in_channels, out_channels, kernel_size=1)` → mezcla solo entre canales.
- `forward` encadena ambas.

### `Pooling.py`
Repaso práctico de capas de agrupación:
- `nn.MaxPool2d(kernel_size=2)` y `nn.AvgPool2d(kernel_size=2)`.
- Comentario: max pooling se usa más que avg porque es más rápido, usa menos memoria, conserva mejor las características importantes y da más invarianza (aunque "pierde" más información).
- **DepthPool**: módulo custom que hace pooling **entre canales** (no espacial) reorganizando dimensiones y usando `F.max_pool1d` — útil para lograr invarianza a otras propiedades (brillo, color...) ya que PyTorch no trae esto de fábrica.
- **Global Average Pooling**: dos formas equivalentes — `nn.AdaptiveAvgPool2d(output_size=1)` o `tensor.mean(dim=(2,3), keepdim=True)`. Reduce cada mapa de características a un único número (destructivo pero útil antes de la capa de salida, como en GoogLeNet/SENet).

---

## 3. Arquitecturas completas de ejemplo

### `Sample_CNN.py`
CNN de ejemplo tipo LeNet para **FashionMNIST**, construida con `nn.Sequential` y `functools.partial` para no repetir `kernel_size=3, padding="same"` (`DefaultConv2d`):
- Patrón: (Conv → ReLU) ×N → MaxPool, repetido 3 veces, duplicando canales de salida en cada bloque (64→128→256) — comentario: los canales se duplican porque, aunque haya pocas características de bajo nivel, hay muchas combinaciones posibles entre ellas.
- Cabeza final: `Flatten` → `Linear` + `ReLU` + `Dropout(0.5)` (×2) → `Linear` de salida a 10 clases.

### `Res_unit.py`
Implementación de **`ResidualUnit`** (bloque residual de ResNet), muy comentado a modo de apunte:
- `DefaultConv2d = partial(nn.Conv2d, kernel_size=3, stride=1, padding=1, bias=False)` (sin bias porque BatchNorm ya lo compensa).
- Rama principal (`main_layers`): Conv → BN → ReLU → Conv → BN (⚠️ importante: **sin ReLU** al final, la activación se aplica *después* de sumar con la rama skip).
- Rama skip (`skip_connection`):
  - Si `stride == 1` (mismas dimensiones): `nn.Identity()`.
  - Si `stride > 1` (cambia tamaño/canales): conv 1×1 con ese stride + BN, para igualar dimensiones antes de sumar.
- `forward`: `F.relu(main_layers(x) + skip_connection(x))`.
- Muy útil como referencia rápida de "por qué no hay ReLU antes de la suma" y "por qué hace falta la conv 1×1 en la rama skip".

### `ResNet-34.py`
Ensambla `ResidualUnit` (de `Res_unit.py`) para construir **ResNet-34** completa, también muy comentado:
- Etapa inicial: `Conv 7×7 stride=2` → `BatchNorm` → `ReLU` → `MaxPool 3×3 stride=2` (224×224 → 56×56).
- Bloques residuales: secuencia `[64]*3 + [128]*4 + [256]*6 + [512]*3` (16 bloques × 2 convs = 32 capas + 1 inicial + 1 FC = **34 capas**). El `stride` de cada bloque es 2 solo cuando cambia el nº de filtros respecto al bloque anterior (para reducir el tamaño espacial), si no, `stride=1`.
- Cabeza: `AdaptiveAvgPool2d(1)` → `Flatten` → `nn.LazyLinear(10)` (infiere automáticamente el nº de entradas, aquí 512).
- Buen archivo para repasar de un vistazo cómo se pasa de "bloque residual" a "arquitectura completa".
- Depende de `Res_unit.py` (`from Res_unit import ResidualUnit`).

---

## 4. Transfer learning y modelos preentrenados

### `Pretrained.py`
Uso más básico de un modelo preentrenado de TorchVision, **sin reentrenar** (solo inferencia):
- Carga pesos con `torchvision.models.ConvNeXt_Base_Weights.IMAGENET1K_V1` y el modelo `convnext_base(weights=pesos)`.
- Importante: usar `pesos.transforms()` para preprocesar igual que en el entrenamiento original.
- `model.eval()` + `torch.no_grad()` para inferencia; `torch.argmax` sobre los logits; nombres de clase en `pesos.meta["categories"]`.
- Menciona cómo listar todos los modelos (`models.list_models`) y todos los pesos disponibles de un modelo concreto (`models.get_model_weights(modelo)`).

### `Transf_pretrained.py`
Caso real de **fine-tuning** sobre el dataset **Flowers102** (102 clases, pocas imágenes por clase):
- Carga `convnext_base` preentrenado + datasets `Flowers102` (train/val/test) usando las transforms del propio modelo.
- **"Cortar la cabeza"**: inspecciona `model.classifier` (LayerNorm → Flatten → Linear 1024→1000) y sustituye la última capa: `model.classifier[2] = nn.Linear(1024, 102)`.
- **Congelar/descongelar**: congela todos los parámetros (`requires_grad = False`) y solo descongela `model.classifier` para entrenar primero la cabeza nueva.
- **Data augmentation** con `torchvision.transforms.v2`: `RandomHorizontalFlip`, `RandomRotation`, `RandomResizedCrop`, `ColorJitter`, seguido de `ToImage`/`ToDtype`/`Normalize` (con las medias/std estándar de ImageNet).
- Entrena con `train_minibatch_gd` (de `ModelUtl.Train`, fuera del alcance de este resumen), `CrossEntropyLoss`, `Adam`, `torchmetrics.Accuracy`.
- Al final deja una lista de ideas para seguir mejorando precisión: probar otros modelos, más datos, ensamblado, descongelar capas gradualmente, learning rates diferenciales por capa, otro optimizador/scheduler, etc. — buena checklist para futuros proyectos de transfer learning.

### `flowerclasses.py`
Simple lista `flowers102_classes` con los 102 nombres de clase del dataset Flowers102 (en inglés), usada como import de apoyo por `Transf_pretrained.py`. No requiere estudio, solo referencia/lookup de nombres de clases.

---

## 5. Clasificación + Localización

### `Class_Local.py`
Implementa `FlowerLocator`: módulo que combina **dos cabezas** sobre un mismo backbone (multi-output):
- `base_model.features` + `avgpool` → produce el vector de características compartido.
- Cabeza 1: `base_model.classifier` (clasificación, ya existente).
- Cabeza 2: `localization_head = Flatten → Linear(in_features, 4)` (nueva, saca 4 números = bounding box).
- `forward` devuelve `(logits, bbox)`.
- Incluye ejemplo de `torchvision.tv_tensors.BoundingBoxes` con formato `"CXCYWH"` (centro x, centro y, ancho, alto) y `canvas_size` (tamaño original de la imagen antes de preprocesar).
- Relacionado directamente con la sección "Clasificación y Localización" del `.md` teórico (para entender IoU/GIoU/CIoU como métrica de esta segunda cabeza).

---

## 6. Ejercicios (`Ejercicios/`)

### `Ejercicios/ex.md`
Respuestas escritas a las preguntas teóricas 1-7 del capítulo:
1. Ventajas de CNN vs DNN completamente conectada.
2. Cálculo numérico completo de **parámetros** y **RAM** de una CNN de 3 capas (a mano, con desglose paso a paso) — muy útil como plantilla para resolver ejercicios similares.
3. 5 trucos si se agota memoria de GPU (batch más pequeño, más stride, usar CPU, quantizar, quitar capas).
4. Por qué max pooling en vez de conv con mismo stride (menos pérdida de info + sin parámetros).
5. (sin responder — innovaciones de AlexNet/GoogLeNet/ResNet/SENet/Xception/EfficientNet/ConvNeXt, ver el `.md` teórico para esto).
6. Qué es una red totalmente convolucional y cómo convertir una capa densa en convolucional.
7. Dificultad técnica principal de la segmentación semántica (pérdida de info espacial) y cómo se mitiga (skip connections).
- Indica que los ejercicios 8, 9 y 10 se resuelven en un notebook Colab (ver siguiente entrada).

### `Ejercicios/Ejercicios-8-9-10.ipynb`
Notebook con la parte práctica (41 celdas):
- **Setup común**: detección de dispositivo (`cuda`/`mps`/`cpu`), función auxiliar `evaluate_tm` para evaluar con `torchmetrics`.
- **Ejercicio 8** (CNN propia para MNIST): define dos variantes — `ModeloMINSTFCN` (totalmente convolucional, termina en `Conv2d` 1×1 para reducir a 10 canales) y `ModeloMINSTDENSE` (termina en capas `Linear`) — entrena ambas con `train_minibatch_gd` y compara el **número de parámetros** entre ambos enfoques (relevante para la pregunta teórica 6 del `.md`).
- **Ejercicio 9** (transfer learning con dataset propio de ≥100 imágenes/clase): las celdas están vacías en este notebook — el propio notebook indica que esto se aplicará en otro repo para mejorar un proyecto antiguo. **Resuelto en:** [github.com/DanielBBHub/ReconocimientoComponentes](https://github.com/DanielBBHub/ReconocimientoComponentes).
- **Ejercicio 10** (tutorial oficial de fine-tuning para detección de objetos, dataset **PennFudan** — peatones):
  - Carga y visualiza imágenes + máscaras de segmentación.
  - Define `PennFudanDataset` (dataset custom que convierte máscaras en bounding boxes con `masks_to_boxes` y usa `tv_tensors`).
  - Dos formas de definir el modelo: (1) fine-tuning directo de `fasterrcnn_resnet50_fpn` preentrenado en COCO, sustituyendo el `FastRCNNPredictor`; (2) cambiar el **backbone** por `mobilenet_v2` + `AnchorGenerator` propio.
  - Modelo final para **detección + segmentación de instancias**: `maskrcnn_resnet50_fpn` con `FastRCNNPredictor` y `MaskRCNNPredictor` reemplazados.
  - Descarga por `wget` los scripts oficiales de referencia de torchvision (`engine.py`, `utils.py`, `coco_utils.py`, `coco_eval.py`, `transforms.py`) — ver siguiente sección.
  - Entrena con `train_one_epoch`/`evaluate` (de `engine.py`) y visualiza predicciones finales con `draw_bounding_boxes`/`draw_segmentation_masks`.

### `Ejercicios/{engine,utils,coco_utils,coco_eval,transforms}.py`
**No son apuntes propios**: son los scripts de referencia oficiales de TorchVision para detección de objetos (`torchvision/references/detection/`), descargados vía `wget` en la celda 35 del notebook de ejercicios (ver arriba). Contienen boilerplate reutilizable:
- `utils.py`: `MetricLogger`, `SmoothedValue`, `collate_fn` para DataLoader con targets de tamaño variable.
- `transforms.py`: transformaciones que actúan sobre `(imagen, target)` a la vez (bounding boxes, keypoints...).
- `coco_utils.py` / `coco_eval.py`: conversión a formato COCO y cálculo de métricas COCO (mAP, etc.) con `pycocotools`.
- `engine.py`: bucles `train_one_epoch` y `evaluate` estándar para modelos de detección.
- Útil únicamente como **referencia de API** si se retoma el ejercicio 10 o un proyecto de detección de objetos; no hace falta estudiarlos línea a línea. Existen copias `.py.1` (backups del propio `wget`) y compilados en `__pycache__/` que se pueden ignorar/borrar sin problema.

---

## 7. Imágenes de apoyo (`img/`)

| Archivo | Contenido |
|---|---|
| `inception_module.png` | Diagrama del módulo Inception (GoogLeNet) |
| `GoogLeNet.png` | Arquitectura completa de GoogLeNet |
| `ResNet.png` | Arquitectura de una unidad residual / ResNet |
| `lista_modelos.png` | Tabla comparativa de modelos preentrenados de PyTorch por tamaño |

Estas imágenes están embebidas directamente en `Deep computer vision using convolutional neural networks.md`.

---

## ⚠️ Notas técnicas transversales detectadas

- `Transf_pretrained.py` y el ejercicio 9 dependen de `ModelUtl.Train.train_minibatch_gd`, módulo excluido de este resumen por instrucción explícita.
