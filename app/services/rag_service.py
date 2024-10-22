import os
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

client = chromadb.Client(path=CHROMA_DB_PATH)

def get_relevant_chunk(question: str) -> str:
    """ Obtiene el chunk más relevante basado en la pregunta del usuario """
    question_embedding = model.encode(question)
    chunk, _ = client.query(question_embedding)
    return chunk