from pathlib import Path
import csv
import random
from datetime import date, timedelta

"""
No existe un dataset público conocido que empareje fechas escritas en formato largo
("April 22, 2019") con su equivalente ISO ("2019-04-22"), así que se genera uno
sintético, tal y como se hace en el ejercicio original: se sortean fechas aleatorias
y se formatean en ambos estilos.
"""

MESES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

def fecha_aleatoria(fecha_min, fecha_max):
    delta_dias = (fecha_max - fecha_min).days
    return fecha_min + timedelta(days=random.randint(0, delta_dias))

def generar_dataset_fechas(n_muestras, seed=42):
    random.seed(seed)
    fecha_min = date(1000, 1, 1)
    fecha_max = date(9999, 12, 31)
    dataset = []
    for _ in range(n_muestras):
        fecha = fecha_aleatoria(fecha_min, fecha_max)
        entrada = f"{MESES[fecha.month - 1]} {fecha.day}, {fecha.year}"
        etiqueta = fecha.strftime("%Y-%m-%d")
        dataset.append((entrada, etiqueta))
    return dataset

def guardar_dataset(dataset, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fecha_escrita", "fecha_iso"])
        writer.writerows(dataset)

if __name__ == "__main__":
    dataset = generar_dataset_fechas(30_000)
    guardar_dataset(dataset, Path(__file__).parent / "dates_dataset.csv")
    print(f"Dataset generado con {len(dataset)} ejemplos en dates_dataset.csv")
    print(f"Ejemplo: {dataset[0]}")
