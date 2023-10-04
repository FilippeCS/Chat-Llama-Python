from llama_cpp import Llama

# Carregar o modelo Llama
print("Carregando o modelo Llama...")
llm = Llama(model_path="./models/ggml-alpaca-7b-q4.bin")
print("Modelo Llama carregado!")

while True:
    # Pergunta para gerar texto
    pergunta = input("Digite uma pergunta (ou digite 'sair' para encerrar): ")

    if pergunta.lower() == "sair":
        break

    # Gere texto usando o modelo Llama
    stream = llm(
        f"Pergunta: {pergunta} Resposta: ",
        max_tokens=100,
        stop=["\n", " Q:"],
        echo=True,
    )

    resultado = next(stream)
    texto_gerado = resultado["choices"][0]["text"]

    # Exiba o texto gerado
    print("Texto gerado:")
    print(texto_gerado)
