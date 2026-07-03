# FINE-TUNING NEURAL NETWORK HYPERPARAMETERS WITH OPTUNA

# Ya se ha revisado como elegir manualmente valores razonables para los hiperparametros del modelo, pero se puede hacer de manera automatizada
# Una opcion es convertir el modelo de PyTorch en un "Estimator" de Scikit-Learn y aplicar GridSearch para los hyperparametros. Otra es utilizar
# una libreria dedicda a configurar estos parametros, como Optuna, Ray Tune o Hyperopt. En este caso utilizaremos Optuna.

# El primer paso es definir una funcion que llame Optuna recurentemente para configurar los hiperparametros, para despues evaluar el modelo y devolver una metrica

import optuna
from ImgClass import ImageClassifier
from ModelUtl.Train import train_minibatch_gd
import torchvision
import torchvision.transforms.v2 as T
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


def objective(trial, train_loader, eval_loader):
    

    # Las funciones suggest_float/int nos permite pedirle a Optuna un valor entre el rango dado
    # En el caso del ratio de aprendijaze, el rango es bastante alto, ademas utilizamos log=True por que desconocemos
    # la escala, con lo que la libreria se encargara de elegir este valor y explorar todas las escalas
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)

    n_hidden = trial.suggest_int("n_hidden", 20, 300)
    model = ImageClassifier(n_inputs=1 * 28 * 28, n_hidden1=n_hidden,
    n_hidden2=n_hidden, n_classes=10).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    xentropy = nn.CrossEntropyLoss()
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
    validation_accuracy = train_minibatch_gd_prune(model, optimizer, xentropy, train_loader, eval_loader, 30, device, trial, accuracy)
    return validation_accuracy

toTensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

# Para empezar con la configuracion creamos un objeto Study y llamamos a su metodo optimize() 
# con la funcion definida antes y un numero de iteraciones como argumentos
train_and_valid_data = torchvision.datasets.FashionMNIST(
root="datasets", train=True, download=True, transform=toTensor)

test_data = torchvision.datasets.FashionMNIST(
root="datasets", train=False, download=True, transform=toTensor)

torch.manual_seed(42)
    
train_data, valid_data = torch.utils.data.random_split(
train_and_valid_data, [55_000, 5_000])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
eval_loader = DataLoader(valid_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)
torch.manual_seed(42)
# Por defecto Optuna utiliza TPE (Tree-structured Parzen Estimator) para la optimizacion de parametros. Es un modelo
# secuencial (aprende de los resultados pasados); Optuna empiez con hiperparametros aleatorios y se dedica a explorar las 
# regiones de parametros mas prometedoras
""" sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction="maximize", sampler=sampler)

objective_with_data = lambda trial: objective(
trial, train_loader=train_loader, eval_loader=eval_loader) """



# Es importante mencionar que tambien puede correr en paralelo en diferentes maquinas, pero hay que implementar una
# BD SQL y definir el parametros de "storage" del metodo create_study(), asi como el nombre study_name y load_if_exists=True

# Por otro lado, mirando los resultados de perdida (si se disparan o no cambian nada con las epocas) puede saberse si un trial es
# bueno o no. Se puede cancelar el entrenamiento y devolver el modelo a su estado original o puede interrumpiurse utilizando la
# excepcion TrialPrunedException, que transmite a Optuna que ignore esa prueba.
# Hay varias clases Pruner implementadas que pueden detectar malos trials, como MedianPruner, que comparara el desempeño
# del modelo con la mediana. 
#   - n_startup_trials: define el momento para empezar la poda
#   - n_warmup_steps: define cuantas iteraciones dejar antes de comparar el desempeño
#   - interval_steps: define cuantas iteraciones esperar entre cada comparacion
""" pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0,
interval_steps=1)
study = optuna.create_study(direction="maximize", sampler=sampler,
pruner=pruner)
study.optimize(objective_with_data, n_trials=5)
print(f"\nMejores parametros del modelo (LR y n_hidden): {study.best_params}")
 """
#-----------------------SAVE AND LOAD-----------------------
from SNL import saveModel, loadModel, saveWeights, loadWeights
import torch.nn.functional as F
model = ImageClassifier(n_inputs=1 * 28 * 28, n_hidden1=166,
n_hidden2=166, n_classes=10).to(device)
learning_rate = 0.08525846
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
xentropy = nn.CrossEntropyLoss()
accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
validation_accuracy = train_minibatch_gd(model, optimizer, xentropy, train_loader, eval_loader, 30, device, accuracy)

saveModel(model, "fashion_mnist")

loaded_model = loadModel("fashion_mnist")
loaded_model.eval()
X_new, y_new = next(iter(eval_loader))
X_new = X_new[0].to(device)
y_pred_logits = model(X_new)
y_top4_logits, y_top4_indices = torch.topk(y_pred_logits, k=4, dim=1)
y_top4_probas = F.softmax(y_top4_logits, dim=1)
print(f"Probabilidades con mas confianza: {y_top4_probas.round(decimals=3)}")

saveWeights(model, "fashion_weights")
loaded_weights = loadWeights("fashion_weights")