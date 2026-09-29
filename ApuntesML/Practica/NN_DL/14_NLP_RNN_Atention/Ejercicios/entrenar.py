import copy
from tqdm import tqdm
import torch
def entrenar_nn(modelo, optimizador, perdida, metrica, entrenamiento, validacion, n_iter, device, scheduler = None, delta = 0.001, paciencia = 10):
    """ Función de entrenamiento para los modelos convolucionales predictivos  """

    mejor_precision = float("-inf")
    mejor_modelo = copy.deepcopy(modelo.state_dict())
    contador_paciencia = 0

    for iteracion in range(n_iter):
        #--------------------- ENTRENAMIENTO --------------------- 
        modelo.train()

        perdida_total = 0.0
        perdida_eval_total = 0.0
        metrica.reset()
        barra = tqdm(entrenamiento, desc=f"Epoca {iteracion}", leave=False)
        for batch in barra:
            # Los DataLoader con secuencias de longitud variable (los que usan
            # el collate_fn con padding) devuelven un tercer elemento con la
            # longitud real de cada secuencia del batch; los datasets de
            # ventana fija (ejercicios 9 y 10) siguen devolviendo solo
            # (muestra, etiqueta), así que aquí soportamos ambos casos
            if len(batch) == 3:
                muestra, etiqueta, longitudes = batch
            else:
                muestra, etiqueta = batch
                longitudes = None

            muestra = muestra.to(device)
            etiqueta = etiqueta.to(device)


            optimizador.zero_grad()

            pred = modelo(muestra, longitudes) if longitudes is not None else modelo(muestra)

            perdida_entrenamiento = perdida(pred, etiqueta)
            perdida_total += perdida_entrenamiento.item()

            perdida_entrenamiento.backward()
            optimizador.step()
            metrica.update(pred, etiqueta)
            metrica_entrenamiento = metrica.compute().item()
            barra.set_postfix(perdida=perdida_entrenamiento.item(), metrica=metrica_entrenamiento)

        precision_eval = eval_entrenamiento(modelo, validacion, metrica, device)
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(precision_eval.item())

            else:
                scheduler.step()

        

        print(f"Epoca {iteracion} / {n_iter}: \nPerdida entrenamiento {perdida_total / len(entrenamiento):.4f} Metrica entrenamiento {metrica_entrenamiento:.4f} \nMetrica validacion {precision_eval.item():.4f}")

        # metrica es una metrica de precision (Accuracy/BinaryAccuracy): cuanto mas
        # alta, mejor, asi que la mejora se mide como un incremento, no un descenso.
        if precision_eval.item() > mejor_precision + delta:
            # Mejora significativa
            mejor_precision = precision_eval.item()
            mejor_modelo = copy.deepcopy(modelo.state_dict())
            contador_paciencia = 0
        else:
            # Sin mejora, o mejora demasiado pequeña para contar
            contador_paciencia += 1
            if contador_paciencia >= paciencia:
                print(f"Early stopping activado en la epoca {iteracion + 1}. Mejor precisión {mejor_precision:.4f}")
                break
    print(f"Entrenamiento acabado. Mejor precisión {mejor_precision:.4f}")
    return mejor_modelo

def eval_entrenamiento(modelo, validacion, metrica, device):
    #--------------------- EVALUACION --------------------- 
    modelo.eval()
    metrica.reset()
    with torch.no_grad():
        for batch in validacion:
            if len(batch) == 3:
                muestra, etiqueta, longitudes = batch
            else:
                muestra, etiqueta = batch
                longitudes = None

            muestra = muestra.to(device)
            etiqueta = etiqueta.to(device)

            pred = modelo(muestra, longitudes) if longitudes is not None else modelo(muestra)
            metrica.update(pred, etiqueta)

    return metrica.compute()
                  
