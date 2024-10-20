from fastapi import APIRouter
from app.services.rag_service import generate_response, search_relevant_chunk
from app.models.questions import Question

router = APIRouter()

@router.post("/ask")
async def ask_question(data: Question):
    relevant_chunk = search_relevant_chunk(data.question)
    response = generate_response(data.question, relevant_chunk)
    return {"user_name": data.user_name, "response": response}