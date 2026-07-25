
# En PyTorch se implementa el recorte de gradiente llamando torch.nn.utils.clip_grad_norm_() o torch.nn.utils.clip_grad_value_()
# justo despues de que los gradientes se hayan calculado
for epoch in range(n_epochs):
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        # La función recibe 2 argumentos: los parametros del modelo cuyos gradientes se han de recortar 
        # y el umbral de recorte 
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()