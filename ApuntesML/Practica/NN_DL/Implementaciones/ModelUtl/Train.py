import torch
import torchmetrics
from ModelUtl.Eval import evaluate, evaluate_tm
# Import de la función de separado de conjuntos de entrenamiento/test
from sklearn.model_selection import train_test_split

def eval_set (model, dataset, device, eval_func = False):

    if not eval_func:
        eval_func = torchmetrics.MeanSquaredError(squared=False).to(device)
        
    tm_eval = evaluate_tm(model, dataset, eval_func, device)
    print(f"\nEvaluacion del modelo con (metrica implementada con torchmetrics): {tm_eval}")

def train_minibatch_gd(model, optimizer, criterion, train_loader, eval_loader, n_epochs, device, eval_func = False):
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

        if not eval_func:
            eval_set(model, eval_loader, device)
        else:
            eval_set(model, eval_loader, device, eval_func)

        # Early stopping sobre val_loss
        if abs(mean_val_loss - last_loss) < early_stopping[0]:
            if early_stopping[1] >= early_stopping[2]:
                print(f"Parada por early stopping con pérdida de validación: {mean_val_loss:.4f}")
                break
            early_stopping[1] += 1
        else:
            last_loss = mean_val_loss
            early_stopping[1] = 0

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
