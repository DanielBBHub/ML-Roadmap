import torch
import torchmetrics
from ModelUtl.Eval import evaluate, evaluate_tm, confusion_matrix_binary_visual
# Import de la función de separado de conjuntos de entrenamiento/test
from sklearn.model_selection import train_test_split
import copy
from torchmetrics.classification import BinaryAccuracy

def eval_set (model, dataset, device, eval_func = False):

    if not eval_func:
        acc = BinaryAccuracy().to(device)
    
    
    tm_eval = evaluate_tm(model, dataset, eval_func, device)
    print("Accuracy:", tm_eval.item())

   

def _prepare_for_loss(y_pred, y_true, criterion):
    """
    Ajusta shapes/dtypes según la loss:
    - BCEWithLogitsLoss: input y target misma forma (normalmente [B,1]), target float
    - CrossEntropyLoss: y_pred [B,C], y_true [B] long
    """
    if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
        # y_pred esperado [B,1] o [B]
        if y_pred.ndim == 2 and y_pred.size(1) == 1 and y_true.ndim == 1:
            y_true = y_true.unsqueeze(1)  # [B] -> [B,1]
        y_true = y_true.float()
    else:
        # Caso típico multiclase
        y_true = y_true.long()

    return y_pred, y_true


def _batch_accuracy_percent(y_pred, y_true, criterion):
    """
    Accuracy en porcentaje para imprimir por época.
    """
    if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
        # Binario con logits
        probs = torch.sigmoid(y_pred)
        preds = (probs >= 0.5).long()

        if y_true.ndim == 1:
            y_true = y_true.unsqueeze(1)
        y_true = y_true.long()

        correct = (preds == y_true).sum().item()
        total = y_true.numel()
    else:
        # Multiclase
        preds = torch.argmax(y_pred, dim=1)
        y_true = y_true.long()
        correct = (preds == y_true).sum().item()
        total = y_true.size(0)

    return correct, total

def train_minibatch_gd(
    model, optimizer, criterion, train_loader, eval_loader,
    n_epochs, device, eval_func=False, min_delta=0.002, patience=6
):
    # Early stopping:
    # min_delta: mejora mínima exigida en val_loss para considerar que realmente mejoró
    # patience: número de épocas consecutivas sin mejora significativa antes de parar
    best_val_loss = float("inf")
    patience_counter = 0

    # Guardamos el mejor estado del modelo para restaurarlo al final
    best_state = copy.deepcopy(model.state_dict())

    eval_calc = None

    for epoch in range(n_epochs):
        # Para diferenciar los diferentes modos de un entrenamiento tenemos model.train() y model.eval()

        # model.train(): Activa comportamiento de entrenamiento:
        #     Dropout: apaga neuronas aleatoriamente.
        #     BatchNorm: usa estadísticas del batch actual y actualiza medias/varianzas internas.
        # Se usa antes del loop de entrenamiento.
        model.train()

        # -------- TRAIN --------
        total_loss = 0.0
        total_train_samples = 0

        for X_batch, y_batch in train_loader:
            # Para mover mas rapido a la GPU los batches, utilizar non_blocking=True
            # para no bloquear el hilo principal
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            # Poner gradientes a cero antes del backward para evitar acumulación
            optimizer.zero_grad()

            y_pred = model(X_batch)

            # Ajustamos shapes/dtypes según la loss:
            # - BCEWithLogitsLoss: y_pred/y_true misma forma (p.ej. [B,1]) y target float
            # - CrossEntropyLoss: y_true tipo long con forma [B]
            y_pred, y_batch_loss = _prepare_for_loss(y_pred, y_batch, criterion)
            loss = criterion(y_pred, y_batch_loss)

            loss.backward()
            optimizer.step()

            # Acumulamos pérdida ponderada por tamaño de batch
            bs = y_batch.size(0)
            total_loss += loss.item() * bs
            total_train_samples += bs

        mean_loss = total_loss / total_train_samples
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {mean_loss:.4f}")

        # model.eval(): Activa comportamiento de inferencia/validación:
        #     Dropout: se desactiva.
        #     BatchNorm: usa estadísticas acumuladas, no las del batch.
        # Se usa para validación/test/inferencia.
        model.eval()

        # Evaluación sobre train para monitorizar (opcional)
        if not eval_func:
            eval_set(model, train_loader, device)
        else:
            eval_set(model, train_loader, device, eval_func)

        # -------- VALIDATION (end of epoch) --------
        val_loss_sum = 0.0
        total_val_samples = 0

        # Para accuracy de validación (%)
        total_correct = 0
        total_items = 0

        # El modo eval no desactiva gradientes, hay que seguir explicitando esta restricción
        with torch.no_grad():
            for X_val, y_val in eval_loader:
                X_val = X_val.to(device, non_blocking=True)
                y_val = y_val.to(device, non_blocking=True)

                y_val_pred = model(X_val)

                # Mismo ajuste de shapes/dtypes que en entrenamiento
                y_val_pred, y_val_loss = _prepare_for_loss(y_val_pred, y_val, criterion)
                vloss = criterion(y_val_pred, y_val_loss)

                # Acumulamos pérdida ponderada por tamaño de batch para cálculo más correcto
                bs = y_val.size(0)
                val_loss_sum += vloss.item() * bs
                total_val_samples += bs

                # Cálculo de acierto para mostrar porcentaje de validación
                c, t = _batch_accuracy_percent(y_val_pred, y_val, criterion)
                total_correct += c
                total_items += t

        mean_val_loss = val_loss_sum / total_val_samples
        val_acc_pct = 100.0 * total_correct / total_items if total_items > 0 else 0.0
        print(f"Epoch {epoch + 1}/{n_epochs}, Val Loss: {mean_val_loss:.4f}, Val Acc: {val_acc_pct:.2f}%")

        if not eval_func:
            eval_calc = eval_set(model, eval_loader, device)
        else:
            eval_calc = eval_set(model, eval_loader, device, eval_func)

        # Early stopping sobre val_loss:
        # Solo consideramos mejora si baja al menos min_delta respecto al mejor valor histórico
        if mean_val_loss < (best_val_loss - min_delta):
            best_val_loss = mean_val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"EarlyStopping: {patience_counter}/{patience} sin mejora significativa")

            if patience_counter >= patience:
                print(f"Parada por early stopping. Mejor val_loss: {best_val_loss:.4f}")
                break
    """ cm = confusion_matrix_binary_visual(
    model,
    eval_loader,
    device,
    threshold=0.5,
    class_names=("Pullover (0)", "T-shirt/top (1)"),
    normalize=False
)
    print("Confusion matrix:\n", cm) """
    # Restauramos el mejor modelo encontrado durante el entrenamiento
    model.load_state_dict(best_state)
    return eval_calc
# El bucle de entrenamiento es igual, pero ahora ya no se trabaja con tensores y autograd directamente,
# si no que los modulos se encargan de hacer ese trabajo.
# Este y el metodo utilizando los tensores, esta calculando "batch gradient descent", es decir, esta calculando
# los gradientes para todo el conjunto de entrenamiento en cada iteración. Si el dataset es pequeño, se puede 
# permitir, pero tiene un problema grande con el escalado
def train_bgd(model, optimizer, criterion, X_train, y_train, n_epochs):
    for epoch in range(n_epochs):
        y_pred = model(X_train)
        # Calculo de la perdida con el modulo recibido
        loss = criterion(y_pred, y_train)
        # Calculo de gradiente de pesos y escalar
        loss.backward()
        # Momento de actualizar pesos y escalar; descenso de gradiente
        optimizer.step()
        # Reestablecer valores a 0 para el gradiente
        optimizer.zero_grad()
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item()}")

def train_test_val(dataset, etiquetas):

    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10
    # train is now 75% of the entire data set
    x_train, x_test, y_train, y_test = train_test_split(dataset, etiquetas, test_size=1 - train_ratio)

    # test is now 10% of the initial data set
    # validation is now 15% of the initial data set
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 


    # Definicion de un tensor con el conjunto de entrenamiento
    x_train = torch.FloatTensor(x_train)
    # Definicion de un tensor con el conjunto de validacion
    x_val = torch.FloatTensor(x_val)
    # Definicion de un tensor con el conjunto de test
    x_test = torch.FloatTensor(x_test)
    # Transformaciones de los conjuntos ----------
    # Se obtienen las medias
    means = x_train.mean(dim=0, keepdims=True)
    # Se obtienen las desviaciones stf
    stds = x_train.std(dim=0, keepdims=True)
    # Se normalizan los valores restandole las medias y dividiendo entre las desviaciones, 
    # calculadas sobre el conjunto de entrenamiento
    x_train = (x_train - means) / stds
    x_val = (x_val - means) / stds
    x_test = (x_test - means) / stds

    # Convertimos los arrays de etiquetas en tensores, ya que las predicciones seran vectores-columna 
    # y los arrays de numpy son vectores unidimensionales, con lo que las redimensionamos añadiendo una dimension
    y_train = torch.FloatTensor(y_train).reshape(-1, 1)
    y_val = torch.FloatTensor(y_val).reshape(-1, 1)
    y_test = torch.FloatTensor(y_test).reshape(-1, 1)

    return x_train,x_test,x_val,y_train,y_test,y_val

def train_minibatch_gd_multiinput(model, optimizer, criterion, train_loader, eval_loader, n_epochs, device):
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
        # Como el dataloador ahora devuelve 3 tensores, modificamos la logica del entrenamiento
        for inputs, y_batch in train_loader:           

            # Para mover mas rapido a la GPU los batches, utilizar non_blocking=true para no bloquear el hilo principal
            inputs = {name: X.to(device) for name, X in inputs.items()}
            y_batch = y_batch.to(device)
            y_pred = model(X_wide=inputs["X_wide"], X_deep=inputs["X_deep"])


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
            for inputs, y_batch in eval_loader:
                inputs = {name: X.to(device) for name, X in inputs.items()}
                y_batch = y_batch.to(device)
                y_pred = model(X_wide=inputs["X_wide"], X_deep=inputs["X_deep"])
                
                vloss = criterion(y_pred, y_batch)
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

def train_minibatch_gd_multoutput(model, optimizer, criterion, train_loader, eval_loader, n_epochs, device):
    early_stopping = [0.05, 0.0, 10.0]
    last_loss = 0
    for epoch in range(n_epochs):
        # Para diferenciar los diferentes modos de un entrenamiento tenemos model.train() y model.eval()

        # model.train(): Activa comportamiento de entrenamiento:

        #     Dropout: apaga neuronas aleatoriamente.
        #     BatchNorm: usa estadísticas del batch actual y actualiza medias/varianzas internas.

        # Se usa antes del loop de entrenamiento.      
        model.train()
        
        # -------- TRAIN --------
        total_loss = 0.0
        # Como el dataloador ahora devuelve 3 tensores, modificamos la logica del entrenamiento
        for inputs, y_batch in train_loader:           

            # Para mover mas rapido a la GPU los batches, utilizar non_blocking=true para no bloquear el hilo principal
            inputs = {name: X.to(device) for name, X in inputs.items()}
            y_batch = y_batch.to(device)
            # Modificacion para obtener las dos predicciones del modelo
            y_pred, y_pred_aux = model(**inputs)
            # Perdida de la respuesta principal de la red
            main_loss = criterion(y_pred, y_batch)
            # Perdida de la respuesta secundaria de la red
            aux_loss = criterion(y_pred_aux, y_batch)
            # Recalculo de la perdida, teniendo en cuenta las dos posibles respuestas
            loss = 0.8 * main_loss + 0.2 * aux_loss
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
            for inputs, y_batch in eval_loader:
                 # Para mover mas rapido a la GPU los batches, utilizar non_blocking=true para no bloquear el hilo principal
                inputs = {name: X.to(device) for name, X in inputs.items()}
                y_batch = y_batch.to(device)
                # Modificacion para obtener las dos predicciones del modelo
                y_pred, y_pred_aux = model(**inputs)
                # Perdida de la respuesta principal de la red
                main_loss = criterion(y_pred, y_batch)
                # Perdida de la respuesta secundaria de la red
                aux_loss = criterion(y_pred_aux, y_batch)
                # Recalculo de la perdida, teniendo en cuenta las dos posibles respuestas
                loss = 0.8 * main_loss + 0.2 * aux_loss
                val_loss += loss.item()

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

def train_minibatch_gd_prune(model, optimizer, criterion, train_loader, eval_loader, n_epochs, device, trial, eval_func = False):
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
        if not eval_func:
            eval_set(model, train_loader, device)
        else:
            eval_set(model, train_loader, device, eval_func)

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
        eval_calc = 0.0

        if not eval_func:
            eval_calc = eval_set(model, eval_loader, device)
        else:
            eval_calc = eval_set(model, eval_loader, device, eval_func)

        trial.report(eval_calc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Early stopping sobre val_loss
        if abs(mean_val_loss - last_loss) < early_stopping[0]:
            if early_stopping[1] >= early_stopping[2]:
                print(f"Parada por early stopping con pérdida de validación: {mean_val_loss:.4f}")
                break
            early_stopping[1] += 1
        else:
            last_loss = mean_val_loss
            early_stopping[1] = 0
    return eval_calc
