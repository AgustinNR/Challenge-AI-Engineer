from chromadb import Client
from langchain_openai import OpenAIEmbeddings
import os

# Instanciar el modelo de embeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

client = Client()

# Cargar documento y generar embeddings
def store_document_embeddings(document_path):
    with open(document_path, "r") as file:
        document_text = file.read()
    
    # Dividir el documento en chunks y generar embeddings
    chunks = [document_text[i:i+500] for i in range(0, len(document_text), 500)]
    for chunk in chunks:
        embedding = model.encode(chunk)
        client.add(chunk, embedding)

# Función para buscar el chunk más relevante
def get_relevant_chunk(question: str) -> str:
    question_embedding = model.encode(question)
    chunk, _ = client.query(question_embedding)
    return chunk