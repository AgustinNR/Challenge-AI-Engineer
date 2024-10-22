from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)

def test_ask_question():
    response = client.post("/ask", json={"user_name": "John Doe", "question": "What is the name of the magical flower?"})
    assert response.status_code == 200
    assert "answer" in response.json()