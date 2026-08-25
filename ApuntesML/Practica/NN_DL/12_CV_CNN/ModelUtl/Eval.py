import torch
import matplotlib.pyplot as plt

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
    if isinstance(x, dict):
        return {k: v.to(device, non_blocking=True) for k, v in x.items()}
    return x.to(device, non_blocking=True)

def evaluate_tm(model, data_loader, metric, device):
    model.eval()
    metric.reset()

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = _move_to_device(X_batch, device)
            y_batch = _move_to_device(y_batch, device)

            # Forward compatible con tensor o dict
            if isinstance(X_batch, dict):
                y_pred, _ = model(X_batch["X_wide"], X_batch["X_deep"])
            else:
                y_pred = model(X_batch)

            # Binario: salida [B,1] o [B] -> sigmoid + threshold
            if y_pred.ndim == 2 and y_pred.size(1) == 1:
                y_pred = torch.sigmoid(y_pred)
                y_pred = (y_pred >= 0.5).int()
                if y_batch.ndim == 1:
                    y_batch = y_batch.unsqueeze(1)
                y_batch = y_batch.int()

            # Multiclase: salida [B,C] -> argmax
            elif y_pred.ndim == 2 and y_pred.size(1) > 1:
                y_pred = torch.argmax(y_pred, dim=1)
                y_batch = y_batch.long()

            # Caso raro [B] binario
            else:
                y_pred = (torch.sigmoid(y_pred) >= 0.5).int()
                y_batch = y_batch.int()

            metric.update(y_pred, y_batch)

    return metric.compute()


def confusion_matrix_binary_visual(
    model,
    data_loader,
    device,
    threshold=0.5,
    class_names=("Pullover (0)", "T-shirt/top (1)"),
    normalize=False,
    figsize=(6, 5),
    cmap="Blues"
):
    """
    Calcula y dibuja matriz de confusión binaria.

    Devuelve:
      cm (torch.Tensor): [[TN, FP], [FN, TP]]
    """
    model.eval()
    tn = fp = fn = tp = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            if isinstance(X_batch, dict):
                X_batch = {k: v.to(device, non_blocking=True) for k, v in X_batch.items()}
                logits, _ = model(X_batch["X_wide"], X_batch["X_deep"])
            else:
                X_batch = X_batch.to(device, non_blocking=True)
                logits = model(X_batch)

            y_batch = y_batch.to(device, non_blocking=True)

            probs = torch.sigmoid(logits)
            if probs.ndim == 1:
                probs = probs.unsqueeze(1)
            if y_batch.ndim == 1:
                y_batch = y_batch.unsqueeze(1)

            y_pred = (probs >= threshold).int()
            y_true = y_batch.int()

            tn += ((y_pred == 0) & (y_true == 0)).sum().item()
            fp += ((y_pred == 1) & (y_true == 0)).sum().item()
            fn += ((y_pred == 0) & (y_true == 1)).sum().item()
            tp += ((y_pred == 1) & (y_true == 1)).sum().item()

    cm = torch.tensor([[tn, fp],
                       [fn, tp]], dtype=torch.float32)

    # Normalización opcional por fila (clase real)
    if normalize:
        row_sums = cm.sum(dim=1, keepdim=True).clamp_min(1e-12)
        cm_plot = cm / row_sums
        fmt = ".2%"
        annot = [[f"{cm_plot[i,j]*100:.2f}%" for j in range(2)] for i in range(2)]
        title = "Matriz de confusión (normalizada por clase real)"
    else:
        cm_plot = cm
        fmt = "g"
        annot = [[f"{int(cm[i,j].item())}" for j in range(2)] for i in range(2)]
        title = "Matriz de confusión (conteos)"

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm_plot.cpu().numpy(),
        annot=annot,
        fmt="",
        cmap=cmap,
        cbar=True,
        xticklabels=[f"Pred: {class_names[0]}", f"Pred: {class_names[1]}"],
        yticklabels=[f"Real: {class_names[0]}", f"Real: {class_names[1]}"]
    )
    plt.title(title)
    plt.xlabel("Clase predicha")
    plt.ylabel("Clase real")
    plt.tight_layout()
    plt.show()

    return cm.int()