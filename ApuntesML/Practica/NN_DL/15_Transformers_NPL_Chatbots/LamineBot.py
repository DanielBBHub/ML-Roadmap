def generate(model, tokenizer, prompt, max_new_tokens=50, **generate_kwargs):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
    pad_token_id=tokenizer.eos_token_id,**generate_kwargs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
# Objetivo: convertir un LLM base (que solo predice el siguiente token) en un chatbot con memoria.
# La clase mantiene en self.context toda la conversación como un único texto con roles ("Yo:" y
# "Lamine:") y, en cada turno, se lo da entero al modelo para que continúe escribiendo la
# respuesta de Lamine. El "recuerdo" no está en el modelo (no tiene estado entre llamadas), sino en
# que el historial completo se vuelve a pasar como prompt cada vez.
lamine_introduction = """
Lamine es un chatbot increible. Sabe de todo y es muy util respondiendo preguntas.
"""
class LamineBot: 
    def __init__(self, model, tokenizer, introduction=lamine_introduction,
    max_answer_length=10_000):
        # introduction: texto inicial que presenta al bot (prompt engineering). Es el comienzo del
        # contexto y condiciona el estilo de todas las respuestas posteriores.
        # max_answer_length: tope de caracteres (no de tokens) de la respuesta, para cortar el
        # bucle de generación si el modelo nunca llega a escribir un nuevo turno de "Yo:".
        self.model = model
        self.tokenizer = tokenizer
        self.context = introduction # historial acumulado: introducción + todos los turnos previos
        self.max_answer_length = max_answer_length

    def chat(self, prompt):
        self.context += "\nYo: " + prompt + "\nLamine:"
        # Se añade el turno del usuario y la etiqueta "Lamine:" al final: así el modelo, que solo
        # sabe continuar texto, "entiende" que lo siguiente que debe escribir es la respuesta de Lamine.
        context = self.context
        start_index = len(context)
        # start_index marca dónde acaba el prompt; todo lo generado a partir de ahí es la respuesta.
        while True:
            extended = generate(self.model, self.tokenizer, context,
            max_new_tokens=100)
            # generate() devuelve el prompt + hasta 100 tokens nuevos (muestreo autorregresivo).
            # Se genera en tandas de 100 porque no sabemos cuánto va a durar la respuesta.
            answer = extended[start_index:]
            # Respuesta acumulada hasta ahora: se recorta el prompt original por posición.
            if ("\nYo: " in answer or extended == context or
                len(answer) >= self.max_answer_length): break
            # Tres condiciones de parada: (1) el modelo empieza a inventar el siguiente turno del
            # usuario ("Yo:"), (2) no ha generado nada nuevo (probablemente token EOS), o (3) la
            # respuesta ya supera el límite de longitud.
            context = extended
            # Si no se ha parado, el texto extendido pasa a ser el nuevo prompt y se sigue generando.
        answer = answer.split("\nYo: ")[0]
        # Se descarta lo que el modelo haya escrito a partir de "Yo:" (turno que no le corresponde).
        self.context += answer
        # La respuesta ya limpia se guarda en el historial para que el siguiente turno la tenga en cuenta.
        return answer.strip()
