# Diferentes implementaciones de programadores de tasa de aprendizaje para 
# optimizar la relación tiempo de entrenamiento - desempeño de modelo

# Exponencial (parametro gamma es una constante, 0.9 por defecto es recomendable)
exp_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
# Al implementar un programador es necesario llamar a .step() al final del entrenamiento
for epoch in range(n_epochs):
    for X_batch, y_batch in train_loader:
        [...] # bucle de entrenamiento
        scheduler.step()

# Coseno (necesario definir el máximo de iteraciones y LR min)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.001)

# Adaptativa
adapt_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.1)
# Es necesario modificar el bucle de entrenamiento para evaluar le métrica en cada iteración y pasar el resultado 
# al programador via .step()
metric = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
for epoch in range(n_epochs):
    for X_batch, y_batch in train_loader:
        [...] # bucle de entrenamiento
        val_metric = evaluate_tm(model, valid_loader, metric).item()
        scheduler.step(val_metric)

# Lineal (parametros a definir para el incremento lineal de LR)
lin_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=3)

# Lineal personalizado (definir tu propia función de incremento)
cust_lin_scheduler = torch.optim.lr_scheduler.LambdaLR( optimizer, lambda epoch: (min(epoch, 3) / 3) * (1.0 - 0.1) + 0.1)

# Como los anteriores, en este caso también hay que modificar el entrenamiento y, en caso de
# tener otros programadores de LR, desactivarlos mientras estos lineales esten funcionando
for epoch in range(n_epochs):
    cust_lin_scheduler.step()
    for X_batch, y_batch in train_loader:
        [...] # bucle de entrenamiento
        if epoch >= 3: # desactivar cualquier otro programador hasta que acabe el de calentamiento
            scheduler.step(val_metric)

# Recocido cosinusoidal con reinicios calientes
cosine_repeat_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
optimizer, T_0=2, T_mult=2, eta_min=0.001)

# Programador de 1 ciclo (parametro a definir el valor máximo de LR)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.01, steps_per_epoch=len(data_loader), epochs=10
)