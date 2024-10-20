
from fastapi import FastAPI
from pydantic import BaseModel
from cachetools import TTLCache
import chromadb
from cohere.classify import CohereClient

app = FastAPI()
cache = TTLCache(maxsize=100, ttl=300)

# Iniciar conexión a ChromaDB
client = chromadb.Client()

class Question(BaseModel):
    user_name: str
    question: str

# Simulación de búsqueda en ChromaDB (asumimos API Key para Cohere)
cohere_client = CohereClient(api_key="YOUR_API_KEY")

def buscar_chunk_relevante(pregunta):
    if pregunta in cache:
        return cache[pregunta]
    
    pregunta_embedding = cohere_client.embed(pregunta)
    chunk = client.query(embedding=pregunta_embedding, n_results=1)
    
    # Si la similaridad es muy baja
    similaridad_minima = 0.7
    if chunk["similarity_score"] < similaridad_minima:
        return "Lo siento, no encontré información suficiente para responder. Intenta reformular la pregunta."
    
    cache[pregunta] = chunk["metadata"]["chunk_text"]
    return chunk["metadata"]["chunk_text"]

def generar_respuesta(pregunta, chunk_relevante):
    max_chunk_length = 300
    if len(chunk_relevante.split()) > max_chunk_length:
        chunk_relevante = " ".join(chunk_relevante.split()[:max_chunk_length]) + "..."
    
    prompt = f"Contexto: {chunk_relevante}\nPregunta: {pregunta}\nResponde en una oración con emojis."
    respuesta = cohere_client.generate(prompt=prompt)
    return respuesta["text"]

@app.post("/ask")
async def ask_question(data: Question):
    chunk_relevante = buscar_chunk_relevante(data.question)
    respuesta = generar_respuesta(data.question, chunk_relevante)
    return {"user_name": data.user_name, "response": respuesta}
