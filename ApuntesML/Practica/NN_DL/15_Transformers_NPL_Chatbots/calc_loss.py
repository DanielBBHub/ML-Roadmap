# Objetivo: calcular la pérdida DPO (Direct Preference Optimization, Ecuación 15-2) para ajustar un LLM
# con preferencias humanas. Para ello hace falta log p(y | x) de la respuesta elegida (y_c) y de la
# rechazada (y_r), tanto con el modelo que se entrena (p_theta) como con el modelo de referencia congelado (p_ref).
import torch
import torch.nn.functional as F


def sum_of_log_probas(model, tokenizer, full_input):
    # full_input: lista de textos "prompt + respuesta" (se procesan todos a la vez, en un batch).
    tokenizer.pad_token = tokenizer.eos_token
    # Mistral no tiene token de padding propio, así que se reutiliza el EOS para igualar longitudes.
    encodings = tokenizer(full_input, return_tensors="pt", padding=True)
    encodings = encodings.to(model.device)
    logits = model(**encodings).logits
    # logits: [batch, n_tokens, vocab_size], un logit por cada token posible del vocabulario en cada posición.

    next_token_ids = encodings.input_ids[:, 1:] # [batch, n_tokens-1]: el token "siguiente" real de cada posición
    next_token_log_probas = -F.cross_entropy(
        logits[:, :-1].permute(0, 2, 1), next_token_ids, reduction="none")
    # Atajo de F.cross_entropy en lugar de log_softmax + torch.gather: devuelve directamente
    # log p(token real siguiente) para cada posición, forma [batch, n_tokens-1].
    # [:, :-1] descarta el último logit (no tiene token siguiente); permute(0, 2, 1) deja la dimensión
    # de clases en segunda posición, como espera cross_entropy; reduction="none" evita que haga la media;
    # el signo menos es porque cross_entropy devuelve la log-verosimilitud negativa.

    padding_mask = encodings.attention_mask[:, :-1]
    # 1 en los tokens reales y 0 en el padding, alineado con next_token_log_probas.
    return (next_token_log_probas * padding_mask).sum(dim=1)
    # Suma de log-probabilidades de toda la secuencia: log p(x y) = log p(x) + log p(y | x), forma [batch].
    # Se usa log p(xy) en vez de log p(y | x) porque el log p(x) extra de la respuesta elegida se cancela
    # con el de la rechazada en la ecuación, y así no hay que localizar dónde empieza la respuesta.


def dpo_loss(model, ref_model, tokenizer, full_input_c, full_input_r, beta=0.1):
    # full_input_c / full_input_r: prompt + respuesta elegida / prompt + respuesta rechazada.
    # beta: controla cuánto se permite que el modelo se aleje del modelo de referencia.
    p_c = sum_of_log_probas(model, tokenizer, full_input_c)
    p_r = sum_of_log_probas(model, tokenizer, full_input_r)
    with torch.no_grad(): # el modelo de referencia está congelado: no se calculan gradientes
        p_ref_c = sum_of_log_probas(ref_model, tokenizer, full_input_c)
        p_ref_r = sum_of_log_probas(ref_model, tokenizer, full_input_r)
    return -F.logsigmoid(beta*((p_c - p_ref_c) - (p_r - p_ref_r))).mean()
    # (p_c - p_ref_c) es delta(y_c) y (p_r - p_ref_r) es delta(y_r) de la ecuación. logsigmoid(z) = log sigma(z)
    # es más estable numéricamente que log(sigmoid(z)). .mean() promedia sobre el batch.
    # Antes de entrenar: model.train() y ref_model.eval().
