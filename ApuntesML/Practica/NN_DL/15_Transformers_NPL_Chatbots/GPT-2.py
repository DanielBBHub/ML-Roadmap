# Imports de la libreria Transformers para descargar el modelo gpt2
# Objetivo: cargar GPT-2 preentrenado (decoder-only) y su tokenizador para generar texto.
from transformers import AutoTokenizer, AutoModelForCausalLM
model_id = "gpt2"  # Checkpoint del Hub de Hugging Face: la versión pequeña de GPT-2
# Cargamos el tokenizador y el modelo
gpt2_tokenizer = AutoTokenizer.from_pretrained(model_id)
# Tokenizador de GPT-2: pasa texto a ids enteros [batch, longitud] (y viceversa con
# decode). Debe ser el mismo con el que se preentrenó el modelo, de ahí el mismo model_id.
gpt2 = AutoModelForCausalLM.from_pretrained(
model_id, device_map="auto", dtype="auto")
# AutoModelForCausalLM devuelve la clase adecuada al checkpoint (aquí GPT2LMHeadModel):
# decoder con atención enmascarada (causal) y una cabeza de modelado de lenguaje que
# da un logit por cada token del vocabulario en cada posición
# ([batch, longitud, vocab_size]) para predecir el siguiente token (autorregresivo).
# device_map="auto": coloca el modelo en el mejor dispositivo (normalmente la GPU);
# con varias GPUs y un modelo que no cabe en una, lo reparte entre ellas (sharding).
# dtype="auto": elige la precisión de los pesos según el checkpoint y el hardware
# (típicamente float16 en una GPU moderna, si no float32). Media precisión usa la mitad
# de memoria, permite cargar modelos mayores y acelera el cómputo, porque la GPU tiene
# aceleración por hardware y se mueven menos datos entre CPU y GPU.

# Función auxiliar: dado un prompt, genera texto de forma autorregresiva (un token por
# paso, realimentando la salida como entrada) y devuelve el texto completo (prompt + continuación).
def generate(model, tokenizer, prompt, max_new_tokens=50, **generate_kwargs):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # Tokeniza el prompt: devuelve un diccionario con input_ids y attention_mask, ambos
    # [1, n_tokens_prompt]. return_tensors="pt" los da como tensores de PyTorch y .to(model.device)
    # los mueve al mismo dispositivo que el modelo (p. ej. la GPU).
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
    pad_token_id=tokenizer.eos_token_id,
    **generate_kwargs)
    # generate(): bucle de decodificación autorregresivo. En cada paso predice el siguiente
    # token a partir de los anteriores y lo añade a la secuencia. Devuelve los ids
    # [1, n_tokens_prompt + n_tokens_nuevos].
    # max_new_tokens: máximo de tokens nuevos a generar (sin contar el prompt).
    # pad_token_id: GPT-2 no tiene token de relleno, así que se reutiliza el de fin de
    # secuencia (eos) para evitar el aviso de generate().
    # **generate_kwargs: permite pasar la estrategia de decodificación (do_sample,
    # temperature, top_k, top_p, num_beams...); por defecto es voraz (greedy, argmax).
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Pasa los ids de la primera (y única) secuencia del batch de vuelta a texto;
    # skip_special_tokens=True omite tokens especiales como el eos.

prompt = "Lionel Messi se ha retirado como jugador profesional de futbol. Aqui están sus declaraciones: "
prompt2 = "Lionel Messi has just retired from professional football. Here is his speech: "
# res = generate(gpt2, gpt2_tokenizer, prompt2, do_sample=True, top_p=0.6)
# Configuración usada: muestreo (do_sample=True) con temperature=2.0 (reparto muy aplanado,
# salidas más diversas) pero limitado a los 3 tokens más probables (top_k=3), lo que evita
# que la alta temperatura dispare tokens absurdos.
# Posibles valores de los argumentos de generate() (rangos orientativos, hay que probar):
# - do_sample: False (por defecto) = voraz/argmax, determinista y propenso a repetirse;
#   True = muestrea del reparto de probabilidades, distinto en cada ejecución.
# - temperature (por defecto 1): <1 (p. ej. 0.7) reparto más concentrado, salidas más
#   predecibles; >1 (p. ej. 1.5-2.0) reparto más plano, salidas más diversas y arriesgadas.
#   Solo tiene efecto con do_sample=True.
# - top_k (en transformers, por defecto 50): solo se muestrea entre los k tokens más
#   probables. Valores bajos (1-10) = conservador (k=1 equivale a voraz); altos (50-100) = más variedad.
# - top_p (por defecto 1.0, desactivado): muestrea del menor conjunto de tokens cuya
#   probabilidad acumulada llega a top_p. Típico 0.9-0.95; más bajo = más conservador.
# - num_beams (por defecto 1, sin beam search): anchura de la búsqueda en haz. Valores
#   como 3-5 exploran varias secuencias a la vez y suelen dar texto más coherente, pero
#   más lento y más propenso a repetir; se usa sin muestreo (do_sample=False).
# - max_new_tokens (aquí 50): longitud máxima de la continuación, sin contar el prompt.
#print(res)

DEFAULT_TEMPLATE = "Capital city of France = Paris\nCapital city of {country} ="
def get_capital_city(model, tokenizer, country, template=DEFAULT_TEMPLATE):
    prompt = template.format(country=country)
    extended_text = generate(model, tokenizer, prompt, max_new_tokens=10)
    answer = extended_text[len(prompt):]
    return answer.strip().splitlines()[0].strip()

cap1 = get_capital_city(gpt2, gpt2_tokenizer, "United Kingdom")
cap2 = get_capital_city(gpt2, gpt2_tokenizer, "Mexico")

#print(cap1)
#print(cap2)