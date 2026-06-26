from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from train_test_val import train_test_val
import multiprocessing as mp

from ModelEval.Eval import evaluate, evaluate_tm
import torchmetrics

def eval_set (model, dataset, device):

    rmse = torchmetrics.MeanSquaredError(squared=False).to(device)
    tm_eval = evaluate_tm(model, dataset, rmse, device)
    print(f"\nEvaluacion del modelo en un TensorDataset con RMSE (metrica implementada con torchmetrics): {tm_eval}")

def train_minibatch_gd(model, optimizer, criterion, train_loader, eval_loader, n_epochs, device):
    early_stopping = [0.05, 0.0, 10.0]
    last_loss = 0
    for epoch in range(n_epochs):
        # Para diferenciar los diferentes modos de un entrenamiento tenemos model.train() y model.eval()

        # model.train(): Activa comportamiento de entrenamiento:

        #     Dropout: apaga neuronas aleatoriamente.
        #     BatchNorm: usa estadísticas del batch actual y actualiza medias/varianzas internas.

        # Se usa antes del loop de entrenamiento.
        model.train()

        
        model.train()
        # -------- TRAIN --------
        total_loss = 0.0
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

        

        # model.eval(): Activa comportamiento de inferencia/validación:

        #     Dropout: se desactiva.
        #     BatchNorm: usa estadísticas acumuladas, no las del batch.

        # Se usa para validación/test/inferencia.

        model.eval()
        eval_set(model, train_loader, device)

        # -------- VALIDATION (end of epoch) --------
        val_loss = 0.0
        # El modo eval no desactiva gradientes, hay que seguir explicitando esta restriccion
        with torch.no_grad():
            for X_val, y_val in eval_loader:
                X_val = X_val.to(device, non_blocking=True)
                y_val = y_val.to(device, non_blocking=True)
                y_val_pred = model(X_val)
                vloss = criterion(y_val_pred, y_val)
                val_loss += vloss.item()

        mean_val_loss = val_loss / len(eval_loader)
        print(f"Epoch {epoch + 1}/{n_epochs}, Val Loss: {mean_val_loss:.4f}")

        eval_set(model, eval_loader, device)

        # Early stopping sobre val_loss
        if abs(mean_val_loss - last_loss) < early_stopping[0]:
            if early_stopping[1] >= early_stopping[2]:
                print(f"Parada por early stopping con pérdida de validación: {mean_val_loss:.4f}")
                break
            early_stopping[1] += 1
        else:
            last_loss = mean_val_loss
            early_stopping[1] = 0


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
    eval_dataset = TensorDataset(x_val, y_val)

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

    eval_loader = DataLoader(
        eval_dataset,
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
    

    # Con el entrenamiento con lotes pequeños en GPu se ha conseguido una perdida notablemente menor que con el GD normal,
    # pero la velocidad de entrenamiento en cada iteracion ha bajado. Para agilizar el entrenamiento podemos hacer dos cosas:
    #   - definir pin_memory=True cuando se crea el data loader, para guardar los lotes en la RAM de la CPU, lo que da acceso
    #     a la DMA para transferirla a la GPU, evitando una copia inecesaria. Posibles problemas pueden ser que el lote no quepa
    #     en la ram de la CPU.
    #   - utilizar prefetch para no esperar a que se procese completamente el lote en la iteracion de entrenamiento, lo que guardaria
    #     en CPU el siguiente lote mientras que la GPU acaba de iterar sobre el actual. Para esta funcionalidad hay que definir un numero de
    #     dataworkers a utilizar en el prefetch, asi como la cantidad de trabajo que tendra cada uno de ellos con prefetch_factor
        # La clase MSELoss tambien es un modulo, asi que dandole las predicciones y las etiquetas calculara el error
    mse = nn.MSELoss()
    train_minibatch_gd(model, optimizer, mse, train_loader, eval_loader, 50, device)

    # Una vez entrenado, para hacer inferencia sobre una muestra, se haria de la siguiente manera
    # Nuevas muestras
    X_new = x_test[:3]
    X_new = X_new.to(device)
    Y_new = y_test[:3]
    Y_new = Y_new.to(device)
    # no_grad() para no tocar los gradientes entrenados
    with torch.no_grad():
        # Prediccion en base a la matriz de caracteristicas multiplicada por los pesos y el escalar
        y_pred = model(X_new)

    for x,y,pred in zip(X_new,Y_new,y_pred):
        print(f"Para la muestra {x} con etiqueta {y.item():.4f} se ha predecido: {pred.item():.4f} |||||||| Diferencia de: {abs(y.item() - pred.item()):.4f}")


    


if __name__ == "__main__":
    # Necesario para evitar errores de multiprocessing al usar DataLoader con num_workers > 0
    mp.freeze_support()
    main()