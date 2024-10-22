from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from app.vector_db.chromadb_handler import get_relevant_chunk
import os

# Cargar la API Key desde el archivo de entorno .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = OpenAI(temperature=0, api_key=OPENAI_API_KEY)


def get_answer(question: str) -> str:
    # Buscar el chunk más relevante desde ChromaDB
    context_chunk = get_relevant_chunk(question)

    # Cargar el template del prompt desde el archivo
    with open("app/prompts_template.txt", "r") as file:
        prompt_template = file.read()

    # Formatear el prompt
    prompt = PromptTemplate(template=prompt_template).format(
        user_question=question,
        context_chunk=context_chunk
    )

    # Obtener la respuesta del LLM
    response = llm(prompt)
    return response.strip()