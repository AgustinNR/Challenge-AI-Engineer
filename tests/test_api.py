import sys
import os

# Añadir el directorio padre al sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_ask_question():
    response = client.post("/ask/", json={"user_name": "John Doe", "question": "What is the name of the magical flower?"})
    assert response.status_code == 200
    assert "response" in response.json()