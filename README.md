
# Challenge AI Engineer 🌟

## Descripción

Este proyecto es una solución simple de tipo RAG (retrieved augmented generation) que permite interactuar con un modelo de lenguaje (LLM) a través de una API. El objetivo es generar respuestas a preguntas formuladas por el usuario sobre un documento específico. 📄 En este caso, la base de conocimiento es un documento que contiene 5 historias cortas que puedes encontrar en: [/data/documento.docx](./data/).

## Objetivo

Desarrollar una API que permita a los usuarios enviar preguntas y recibir respuestas basadas en un contexto proporcionado por un documento. 🤖✨

## Componentes

1. **API**: Desarrollada en Python utilizando [FastAPI](https://fastapi.tiangolo.com/), que funciona como intermediario entre el usuario y el LLM.
   
   **Estructura del Request**:
   ```json
   {
       "user_name": "John Doe",
       "question": "How are you today?"
   }
   ```

3. **Framework**: Utiliza [Langchain](https://python.langchain.com/docs/introduction/) 🦜🔗, una biblioteca que facilita la creación de aplicaciones de inteligencia artificial. Permite integrar modelos de lenguaje, gestionar flujos de trabajo y conectar con bases de datos y APIs. Además, ofrece herramientas para construir prompts dinámicos y encadenar llamadas a modelos, simplificando el desarrollo y mejorando la escalabilidad.

3. **LLM**: Se comunica con el LLM mediante la API de OpenAI, específicamente utiliza el modelo [gpt-4o](https://platform.openai.com/docs/models/gpt-4o), para responder a las preguntas del usuario. 🌐

4. **Embeddings**: El documento se divide en "chunks" y mediante el modelo [text-embedding-3-small](https://platform.openai.com/docs/guides/embeddings/embedding-models) de OpenAI, estos se vectorizan y almacenan en una base de datos vectorial de ChromaDB. Al recibir una pregunta, se codifica y busca el chunk más relevante para proporcionar contexto al LLM. 📚

5. **Prompt**: El prompt enviado al LLM incluye la pregunta del usuario, el contexto y cualquier otro elemento necesario para cumplir con los requisitos de respuesta.

## Requisitos de Respuesta para el Challenge

- Respuestas consistentes para la misma pregunta. 🔄
- Respuestas en una sola oración. ✍️
- Idioma de respuesta igual al de la pregunta. 🌍
- Inclusión de emojis que resuman el contenido. 😄
- Respuestas siempre en tercera persona. 🗣️

## Configuración del archivo .env 🛠️

Asegúrate de que tu archivo `.env` contenga las siguientes variables:

```bash
OPENAI_API_KEY=<your_api_key>

LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY=<your_api_key>
LANGCHAIN_PROJECT="ai-challenge"
```


Reemplaza `<your_api_key>` con tus claves de API correspondientes. La clave de [**OpenAI**](https://platform.openai.com/docs/overview) es esencial para que la aplicación funcione correctamente, mientras que la clave de [**LangSmith**](https://www.langchain.com/langsmith) es interesante para ver el trazado de las llamadas y mejorar la supervisión del rendimiento de las interacciones con el modelo.


## Instalación (sin Docker) 🚀

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

4. Accede a la documentación de la API en `http://localhost:7000/docs` para probar los endpoints. 📖

## Docker 🐳

Para ejecutar la API en un contenedor [Docker](https://docs.docker.com/), se proporciona un Dockerfile. Sigue estos pasos:

1. Construye la imagen:
   ```bash
   docker build -t rag-challenge .
   ```

2. Ejecuta el contenedor:
   ```bash
   docker run -p 7000:7000 rag-challenge
   ```


## Pruebas 🧪

Una vez instalada y ejecutada la API, puedes acceder al Swagger de la API en `http://localhost:7000/docs` para probar los endpoints. Por ejemplo, prueba las siguientes preguntas:
- ¿Quién es Zara?
- What did Emma decide to do? 
- What is the name of the magical flower? 

