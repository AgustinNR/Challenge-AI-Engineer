from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from document_chunker import run_llm

app = FastAPI()

class UserQuestion(BaseModel):
    user_name: str
    question: str

@app.post("/ask/")
async def ask_question(query: UserQuestion):
    try:
        chat_history = []  # Histórico de chats, si hay
        response = run_llm(query.question, chat_history)
        return {"response": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)