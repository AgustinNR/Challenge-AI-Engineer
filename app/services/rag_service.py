from app.db.vector_db import client
import cohere
from cachetools import TTLCache
from app.config.settings import settings

co = cohere.ClientV2(api_key=settings.cohere_key)
cache = TTLCache(maxsize=100, ttl=300)

def search_relevant_chunk(question):
    if question in cache:
        return cache[question]
    
    question_embedding = co.embed(question)
    chunk = client.query(embedding=question_embedding, n_results=1)
    
    # Si la similaridad es muy baja
    similaridad_minima = 0.7
    if chunk["similarity_score"] < similaridad_minima:
        return "Lo siento, no encontré información suficiente para responder. Intenta reformular la question."
    
    cache[question] = chunk["metadata"]["chunk_text"]
    return chunk["metadata"]["chunk_text"]

def generate_response(question, relevant_chunk):
    prompt = f"Contexto: {relevant_chunk}\\nquestion: {question}\\nResponde en una oración con emojis."
    respuesta = co.generate(prompt=prompt)
    return respuesta["text"]