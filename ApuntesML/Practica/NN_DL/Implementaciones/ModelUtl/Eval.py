import torch

# Funcion que recibe un modelo, un dataloader y una funcion para calcular el desempeño del modelo
def evaluate(model, data_loader, metric_fn, device, aggregate_fn=torch.mean):
    model.eval()
    metrics = []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            metric = metric_fn(y_pred, y_batch)
            metrics.append(metric)
    return aggregate_fn(torch.stack(metrics))


def _move_to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_move_to_device(v, device) for v in x)
    return x  # por si aparece otro tipo


# En vez de implementar metricas manualmente, podemos utilizar Torchmetrics, 
# una libreria desarrollada por el mismo equipo de Pytorch Lightning que tiene
# implementadas varias "streaming metrics". Estas "streaming metrics" son 
# metricas que pueden seguir una metrica y ser actualizadas en cada lote procesado
def evaluate_tm(model, data_loader, metric, device):
    model.eval()
    metric.reset()
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = _move_to_device(X_batch, device)
            y_batch = _move_to_device(y_batch, device)

            # model(X_batch) funciona si:
            # - X_batch es tensor, o
            # - tu forward acepta dict (forward(self, X) con X["X_wide"], X["X_deep"])
            if isinstance(X_batch, dict):
                
                y_pred, _ = model(X_batch["X_wide"], X_batch["X_deep"])
          
            else:
                y_pred = model(X_batch)

            metric.update(y_pred, y_batch)
    return metric.compute()
