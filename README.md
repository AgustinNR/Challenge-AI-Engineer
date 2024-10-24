
# Challenge AI Engineer

## Descripción

Este proyecto es una solución simple de tipo RAG (retrieved augmented generation) que permite interactuar con un modelo de lenguaje (LLM) a través de una API. El objetivo es generar respuestas a preguntas formuladas por el usuario sobre un documento específico.

## Objetivo

Desarrollar una API que permita a los usuarios enviar preguntas y recibir respuestas basadas en un contexto proporcionado por un documento.

## Componentes

1. **API**: Desarrollada en Python utilizando FastAPI, que funciona como intermediario entre el usuario y el LLM.
   
   **Estructura del Request**:
   ```json
   {
       "user_name": "John Doe",
       "question": "How are you today?"
   }
   ```

2. **LLM**: Utiliza una API de LLM (como OpenAI o Cohere) para responder a las preguntas del usuario.

3. **Embeddings**: El documento se divide en "chunks" para ser codificados y almacenados en una base de datos vectorial (ej. ChromaDB). Al recibir una pregunta, se codifica y busca el chunk más relevante para proporcionar contexto al LLM.

4. **Prompt**: El prompt enviado al LLM incluye la pregunta del usuario, el contexto y cualquier otro elemento necesario para cumplir con los requisitos de respuesta.

## Requisitos de Respuesta

- Respuestas consistentes para la misma pregunta.
- Respuestas en una sola oración.
- Idioma de respuesta igual al de la pregunta.
- Inclusión de emojis que resuman el contenido.
- Respuestas siempre en tercera persona.

## Pruebas

El sistema debe responder correctamente a preguntas como:
- ¿Quién es Zara?
- What did Emma decide to do?
- What is the name of the magical flower?

## Instalación

1. Clona este repositorio:
   ```bash
   git clone https://github.com/AgustinNR/Challenge-AI-Engineer.git
   cd Challenge-AI-Engineer
   ```

2. Crea un entorno virtual e instálalo:
   ```bash
   python -m venv env
   source env/bin/activate  # En Windows usa `env\Scripts\activate`
   pip install -r requirements.txt
   ```

3. Ejecuta la API:
   ```bash
   uvicorn main:app --reload
   ```

4. Accede a la documentación de la API en `http://localhost:8000/docs` para probar los endpoints.

## Docker

Para ejecutar la API en un contenedor Docker, se proporciona un Dockerfile. Sigue estos pasos:

1. Construye la imagen:
   ```bash
   docker build -t ai-engineer .
   ```

2. Ejecuta el contenedor:
   ```bash
   docker run -p 8000:8000 ai-engineer
   ```

## Contribuciones

Las contribuciones son bienvenidas. Si deseas colaborar, por favor abre un issue o envía un pull request.

## Licencia

Este proyecto está licenciado bajo la [MIT License](LICENSE).
