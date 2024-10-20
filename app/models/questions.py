from pydantic import BaseModel

class Question(BaseModel):
    user_name: str   # Nombre del usuario que hace la pregunta.
    question: str    # Pregunta que el usuario hace.