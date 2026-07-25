import torch
import torch.nn as nn

def extract_architecture(model):
    layers = []
    for layer in model:
        if isinstance(layer, nn.Flatten):
            layers.append({
                "type": "Flatten",
                "start_dim": layer.start_dim,
                "end_dim": layer.end_dim
            })
        elif isinstance(layer, nn.Linear):
            layers.append({
                "type": "Linear",
                "in_features": layer.in_features,
                "out_features": layer.out_features,
                "bias": layer.bias is not None
            })
        elif isinstance(layer, nn.BatchNorm1d):
            layers.append({
                "type": "BatchNorm1d",
                "num_features": layer.num_features,
                "eps": layer.eps,
                "momentum": layer.momentum,
                "affine": layer.affine,
                "track_running_stats": layer.track_running_stats
            })
        elif isinstance(layer, nn.ReLU):
            layers.append({"type": "ReLU"})
        else:
            raise ValueError(f"Capa no soportada: {type(layer)}")
    return layers

def build_model_from_architecture(architecture):
    layers = []
    for cfg in architecture:
        t = cfg["type"]

        if t == "Flatten":
            layers.append(nn.Flatten(
                start_dim=cfg.get("start_dim", 1),
                end_dim=cfg.get("end_dim", -1)
            ))

        elif t == "Linear":
            layers.append(nn.Linear(
                in_features=cfg["in_features"],
                out_features=cfg["out_features"],
                bias=cfg.get("bias", True)
            ))

        elif t == "BatchNorm1d":
            layers.append(nn.BatchNorm1d(
                num_features=cfg["num_features"],
                eps=cfg.get("eps", 1e-5),
                momentum=cfg.get("momentum", 0.1),
                affine=cfg.get("affine", True),
                track_running_stats=cfg.get("track_running_stats", True)
            ))

        elif t == "ReLU":
            layers.append(nn.ReLU())

        else:
            raise ValueError(f"Tipo de capa desconocido: {t}")

    return nn.Sequential(*layers)

def saveWeights(model, name):
    model_data = {
        "model_state_dict": model.state_dict(),
        "model_hyperparameters": {
            "architecture": extract_architecture(model)
        }
    }
    torch.save(model_data, "models/" + name + ".pt")

def loadWeights(name, map_location=None, eval_mode=True):
    """
    name: nombre del archivo .pt
    map_location: 'cpu', 'cuda', etc. (opcional)
    eval_mode: si True, deja el modelo en .eval()
    """
    checkpoint = torch.load("models/" + name + ".pt", map_location=map_location)

    if "model_hyperparameters" not in checkpoint:
        raise KeyError("No se encontró 'model_hyperparameters' en el checkpoint.")
    if "architecture" not in checkpoint["model_hyperparameters"]:
        raise KeyError("No se encontró 'architecture' dentro de 'model_hyperparameters'.")
    if "model_state_dict" not in checkpoint:
        raise KeyError("No se encontró 'model_state_dict' en el checkpoint.")

    architecture = checkpoint["model_hyperparameters"]["architecture"]
    model = build_model_from_architecture(architecture)

    model.load_state_dict(checkpoint["model_state_dict"])

    if eval_mode:
        model.eval()

    return model