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


# En vez de implementar metricas manualmente, podemos utilizar Torchmetrics, 
# una libreria desarrollada por el mismo equipo de Pytorch Lightning que tiene
# implementadas varias "streaming metrics". Estas "streaming metrics" son 
# metricas que pueden seguir una metrica y ser actualizadas en cada lote procesado
def evaluate_tm(model, data_loader, metric, device):
    model.eval()
    metric.reset() # reset the metric at the beginning
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            metric.update(y_pred, y_batch) # update it at each iteration
    return metric.compute() # compute the final result at the end