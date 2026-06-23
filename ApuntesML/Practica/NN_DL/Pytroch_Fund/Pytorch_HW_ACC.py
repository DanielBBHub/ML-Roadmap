# El primer paso es el de comprobar que la grafica esta soportada para el HW ACC
""" 
import torch

print("PyTorch version:", torch.__version__)

print("\n=== CUDA ===")
print("Built with CUDA:", torch.backends.cuda.is_built()) ----> El compilado de PyTorch tiene soporte de CUDA
print("CUDA available:", torch.cuda.is_available()) ----> Tienes una o mas graficas compatibles con CUDA

if torch.cuda.is_available(): ----> Informacion sobre las graficas compatibles con CUDA
    print("GPU count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("GPU name:", torch.cuda.get_device_name(0))

"""
import torch
# Comprobacion en tiempo de ejecucion para el soporte de CUDA
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Definicion de un tensor y copia de este a la GPU [to(device)]
M = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
M = M.to(device)

# Definicion directa de un tensor en grafica
M2 = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device=device)

# Comprobacion del dispositivo donde esta x tensor
print(f"\nComprobacion donde esta el tensor M: {M.device}")
print(f"\nComprobacion donde esta el tensor M2: {M2.device}")

# Calculo de multiplicacion de matrices en GPU, ya que esta definido el tensor en la GPU. Esto es una ventaja ya 
# que no necesitamos pasar informacion entre CPU y GPU, ahorrando tiempo y capacidad de computacion
R = M @ M.T
print(f"\nResultado de la multiplicacion de matrices en GPU: {R}")

import time
# Comprobacion del tiempo usado en calcular la misma operacion en CPU vs GPU
M = torch.rand((1000, 1000)) # on the CPU
inicio = time.perf_counter()
resultado = M @ M.T
fin = time.perf_counter()
# En mi dispositivo: Multiplicación en CPU: 0.004541 s
res_cpu = fin - inicio
print(f"\nMultiplicación en CPU: {res_cpu:.6f} s")

M = torch.rand((1000, 1000), device="cuda") # on the GPU
torch.cuda.synchronize()
inicio = time.perf_counter()
R = M @ M.T
torch.cuda.synchronize()
fin = time.perf_counter()
res_gpu = fin - inicio
# En mi dispositivo: Multiplicación en GPU: 0.002645 s
print(f"\nMultiplicación en GPU: {fin - inicio:.6f} s")

print(f"\nDiferencia entre CPU y GPU: {res_cpu - res_gpu:.6f}.En absoluto unas {res_cpu/res_gpu:.3f} veces")