# Ejercicios

1. ¿Cuál es la capa más importante de la arquitectura Transformer? ¿Cuál es su propósito?
2. ¿Por qué la arquitectura Transformer necesita codificaciones posicionales (*positional encodings*)?
3. ¿En qué tareas son mejores los modelos de solo codificador (*encoder-only*)? ¿Y los de solo decodificador (*decoder-only*)? ¿Y los de codificador-decodificador?
4. ¿Cuál es la técnica más importante utilizada para preentrenar BERT?
5. ¿Puedes nombrar cuatro variantes de BERT y explicar sus principales ventajas?
6. ¿Cuál es la tarea principal utilizada para preentrenar GPT y sus sucesores?
7. El método `generate()` tiene muchos argumentos, entre ellos `do_sample`, `top_k`, `top_p`, `temperature` y `num_beams`. ¿Qué hace cada uno de estos cinco argumentos?
8. ¿Qué es la ingeniería de prompts (*prompt engineering*)? ¿Puedes describir cinco técnicas de ingeniería de prompts?
9. ¿Cuáles son los pasos principales para construir un chatbot a partir de un modelo preentrenado de solo decodificador?
10. ¿Cómo puede un chatbot usar herramientas como una calculadora o una búsqueda web?
11. ¿Para qué se usa MCP?
12. Ajusta (*fine-tune*) BERT para análisis de sentimientos con el dataset IMDb.
13. Ajusta GPT-2 con el dataset de Shakespeare (del capítulo 14) y luego genera texto al estilo de Shakespeare.
14. Descarga el dataset Wikipedia Movie Plots y usa SBERT para generar el embedding de la descripción de cada película. Después, escribe una función que reciba una consulta de búsqueda, calcule su embedding, encuentre los embeddings más similares y liste las películas correspondientes.
15. Usa un modelo ajustado para instrucciones, como Qwen-7B-Instruct, para construir un pequeño chatbot que actúe como experto en cine. Después, prueba a añadirle funcionalidad RAG, por ejemplo inyectando automáticamente en el prompt la sinopsis más relevante (véase el ejercicio anterior).
