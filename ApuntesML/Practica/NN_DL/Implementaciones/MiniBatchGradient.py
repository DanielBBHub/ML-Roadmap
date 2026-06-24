from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from train_test_val import train_test_val
import multiprocessing as mp


def train_minibatch_gd(model, optimizer, criterion, train_loader, n_epochs, device):
    early_stopping = [0.05, 0.0, 10.0]
    last_loss = 0
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.
        for X_batch, y_batch in train_loader:
            # Para mover mas rapido a la GPU los batches, utilizar non_blocking=true para no bloquear el hilo principal
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        mean_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {mean_loss:.4f}")

        """ if abs(mean_loss - last_loss) < early_stopping[0]:
            if early_stopping[1] == early_stopping[2]:
                print(f"Parada por early stopping con perdida: {mean_loss}")
                break
            else:
                early_stopping[1] += 1
        else:
            last_loss = mean_loss
            early_stopping[1] = 0 """


def main():
    # Comprobacion en tiempo de ejecucion para el soporte de CUDA
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    housing = fetch_california_housing()

    x_train, x_test, x_val, y_train, y_test, y_val = train_test_val(housing.data, housing.target)

    torch.manual_seed(42)

    # DataLoader es una clase implementada con el objetivo de realizar el descenso de gradiente en lotes pequeños,
    # con un tamaño predefinido, incluso mezclando la informacion en cada iteracion.
    # Es importante resaltar que esta clase espera un objeto dataset que cuente con __len__(self) y __getitem__(self)
    # para obtener el numero de muestras y las muestras, respectivamente
    train_dataset = TensorDataset(x_train, y_train)

    num_workers = 4 if device in ("cuda", "mps") else 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        pin_memory=(device == "cuda"),      # pin_memory útil sobre todo en CUDA
        num_workers=num_workers,             # workers para carga en paralelo
        prefetch_factor=2 if num_workers > 0 else None,  # lotes prefetched por worker
        persistent_workers=(num_workers > 0) # evita recrear workers cada epoch
    )

    n_features = x_train.shape[1]
    # Pytorch tiene un modulo Sequential que utiliza ejecuta de manera secuencial la predicción sobre los inputs
    # y utiliza la salida de estos como entrada de los siguientes [Input -> Linear -> ReLU -> Linear -> Relu -> Linear -> Pred]
    model = nn.Sequential(
        # Capa de entrada con n neuronas de entrada; tantas como caracteristicas en los inputs y 50 salidas
        nn.Linear(n_features, 50),
        # Funcion de activacion para la primera capa intermedia
        nn.ReLU(),
        # Tantas entradas como salidas en el modelo anterior y n salidas
        nn.Linear(50, 40),
        # Funcion de activacion para la primera capa intermedia
        nn.ReLU(),
        # Tantas entradas como salidas en el modelo anterior y 1 salida para la prediccion
        nn.Linear(40, 1)
    )
    model = model.to(device)

    # Antes con lr = 0.1 acababa dando nan la perdida, hay que tener en cuenta lr pequeños
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # La clase MSELoss tambien es un modulo, asi que dandole las predicciones y las etiquetas calculara el error
    mse = nn.MSELoss()

    # Con el entrenamiento con lotes pequeños en GPu se ha conseguido una perdida notablemente menor que con el GD normal,
    # pero la velocidad de entrenamiento en cada iteracion ha bajado. Para agilizar el entrenamiento podemos hacer dos cosas:
    #   - definir pin_memory=True cuando se crea el data loader, para guardar los lotes en la RAM de la CPU, lo que da acceso
    #     a la DMA para transferirla a la GPU, evitando una copia inecesaria. Posibles problemas pueden ser que el lote no quepa
    #     en la ram de la CPU.
    #   - utilizar prefetch para no esperar a que se procese completamente el lote en la iteracion de entrenamiento, lo que guardaria
    #     en CPU el siguiente lote mientras que la GPU acaba de iterar sobre el actual. Para esta funcionalidad hay que definir un numero de
    #     dataworkers a utilizar en el prefetch, asi como la cantidad de trabajo que tendra cada uno de ellos con prefetch_factor
    train_minibatch_gd(model, optimizer, mse, train_loader, 50, device)


if __name__ == "__main__":
    # Necesario para evitar errores de multiprocessing al usar DataLoader con num_workers > 0
    mp.freeze_support()
    main()