
import torch.nn as nn

import torch
import multiprocessing as mp
from WideNDeep import WideAndDeep, WideAndDeepV2, WideAndDeepV3, WideAndDeepV4, WideAndDeepDataset

def main():
    # Comprobacion en tiempo de ejecucion para el soporte de CUDA
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    from ModelUtl.Train import train_minibatch_gd, train_test_val, train_minibatch_gd_multiinput, train_minibatch_gd_multoutput
    from ModelUtl.Eval import evaluate_tm
    from sklearn.datasets import fetch_california_housing
    from torch.utils.data import TensorDataset, DataLoader

    housing = fetch_california_housing()

    x_train, x_test, x_val, y_train, y_test, y_val = train_test_val(housing.data, housing.target)

    train_dataset = TensorDataset(x_train, y_train)
    eval_dataset = TensorDataset(x_val, y_val)
    train_data_wd = WideAndDeepDataset(x_train[:, :5], x_train[:, 2:], y_train)
    eval_dataset_wd = WideAndDeepDataset(x_val[:, :5], x_val[:, 2:], y_val)

    torch.manual_seed(42)
    n_features = x_train.shape[1]
    model = WideAndDeep(n_features).to(device)
    model2 = WideAndDeepV2(n_features).to(device)
    model3 = WideAndDeepV3(n_features).to(device)
    model4 = WideAndDeepV4(n_features).to(device)
    learning_rate = 0.002 

    

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
    
    train_loader_wd = DataLoader(
        train_data_wd,
        batch_size=32,
        shuffle=True,
        pin_memory=(device == "cuda"),      # pin_memory útil sobre todo en CUDA
        num_workers=num_workers,             # workers para carga en paralelo
        prefetch_factor=2 if num_workers > 0 else None,  # lotes prefetched por worker
        persistent_workers=(num_workers > 0) # evita recrear workers cada epoch
    )
    eval_loader_wd = DataLoader(
        eval_dataset_wd,
        batch_size=32,
        shuffle=True,
        pin_memory=(device == "cuda"),      # pin_memory útil sobre todo en CUDA
        num_workers=num_workers,             # workers para carga en paralelo
        prefetch_factor=2 if num_workers > 0 else None,  # lotes prefetched por worker
        persistent_workers=(num_workers > 0) # evita recrear workers cada epoch
    )

    # Un optimizador por modelo, para que cada uno tenga conocimiento sobre los parametros de cada uno de ellos
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate)
    optimizer3 = torch.optim.Adam(model3.parameters(), lr=learning_rate)
    optimizer4 = torch.optim.Adam(model4.parameters(), lr=learning_rate)
    mse = nn.MSELoss()

    # train_minibatch_gd(model, optimizer, mse, train_loader, eval_loader, 50, device)
    # train_minibatch_gd(model2, optimizer2, mse, train_loader, eval_loader, 50, device)
    # train_minibatch_gd_multiinput(model3, optimizer3, mse, train_loader_wd, eval_loader_wd, 50, device)
    train_minibatch_gd_multoutput(model4, optimizer4, mse, train_loader_wd, eval_loader_wd, 50, device)
        # Una vez entrenado, para hacer inferencia sobre una muestra, se haria de la siguiente manera
    # Nuevas muestras
    X_new = x_test[:3].to(device)   
    Y_new = y_test[2:]

    # Nuevas muestras para el modelo multiinput
    X_new_wide = X_new[:, :5]       
    X_new_deep = X_new[:, 2:]       
    
    X_new = X_new.to(device)
    X_new_wide = X_new_wide.to(device)       
    X_new_deep = X_new_deep.to(device) 
    
    Y_new = Y_new.to(device)
    # no_grad() para no tocar los gradientes entrenados
    with torch.no_grad():
        # Prediccion en base a la matriz de caracteristicas multiplicada por los pesos y el escalar
        # y_pred = model(X_new)
        # y_pred2 = model2(X_new)
        # y_pred3 = model3(X_new_wide, X_new_deep)
        y_pred, y_pred_aux = model4(X_new_wide, X_new_deep)

    # for x,y,pred,pred2, pred3 in zip(X_new,Y_new,y_pred, y_pred2, y_pred3):
    #     print(f"Para la muestra con etiqueta {y.item():.4f} se ha predecido: {pred.item():.4f} [-] {pred2.item():.4f} [-] {pred3.item():.4f}|||||||| Diferencia de: {abs(y.item() - pred.item()):.4f} [-] {abs(y.item() - pred2.item()):.4f} [-] {abs(y.item() - pred3.item()):.4f}")

    for main_pred, aux_pred, y in zip(y_pred, y_pred_aux, Y_new):
        print(f"Para la muestra con etiqueta {y.item():.4f} se han predecido: {main_pred.item():.4f} [-] {aux_pred.item():.4f} |||||||| Diferencia de: {abs(y.item() - main_pred.item()):.4f} [-] {abs(y.item() - aux_pred.item()):.4f}")

if __name__ == "__main__":
    # Necesario para evitar errores de multiprocessing al usar DataLoader con num_workers > 0
    mp.freeze_support()
    main()

