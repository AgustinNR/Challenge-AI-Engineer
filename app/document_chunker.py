from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import List, Dict, Any
from dotenv import load_dotenv
import os
from pathlib import Path
import getpass

# Ruta base del proyecto (carpeta padre de 'app')
BASE_DIR = Path.cwd()

# Definir la ruta hacia el archivo .env
ENV_PATH = BASE_DIR / '.env'

# Check if the .env file actually exists
if not ENV_PATH.exists():
    raise FileNotFoundError(f"The .env file does not exist at: {ENV_PATH}")

# Cargar explícitamente el archivo .env
load_dotenv(dotenv_path=ENV_PATH)

# Cargar la API Key desde el archivo de entorno .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

data_dir = "./data/documento.docx"
persist_directory = "./db_index"

# Definir modelo de embeddings
emb = OpenAIEmbeddings(model="text-embedding-3-small")

if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

def load_or_create_vectorstore():
    # Verificar si la base de datos de vectores ya existe
    if not Path(persist_directory).exists():
        # Si no existe, se debe crear la base de datos con los embeddings
        loader = Docx2txtLoader(data_dir)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )
        docs = text_splitter.split_documents(data)
        print(f"Número de fragmentos: {len(docs)}")

        # Crear la base de datos de vectores persistente
        vectorstore = Chroma.from_documents(documents=docs,
                                            embedding=emb,
                                            persist_directory=persist_directory)
    else:
        # Si ya existe, cargar la base de datos de vectores
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=emb
        )
    
    return vectorstore


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Definir la función para ejecutar el LLM con RAG utilizando Chroma
def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    # Cargar el vectorstore persistente de Chrom
    docsearch = load_or_create_vectorstore()
    
    # Inicializar el LLM (ChatOpenAI) para la generación de respuestas
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", verbose=True, temperature=0)
 
    # Crear un prompt personalizado con las instrucciones deseadas
    custom_prompt = PromptTemplate(
        input_variables=["context", "input"],
        template=(
            "Contexto relevante: {context}\n"
            "Pregunta: {input}\n\n"
            "Responde la siguiente pregunta respetando las siguientes reglas:\n"
            "- Responde en solo una oración.\n"
            "- Usa el mismo idioma en el que se hizo la pregunta.\n"
            "- Agrega emojis que resuman el contenido de la respuesta.\n"
            "- Responde siempre en tercera persona.\n\n"
            "Respuesta:"
        )
    )

    # Crear un parser para convertir la respuesta del LLM en cadena de texto
    output_parser = StrOutputParser()

    # Buscar los documentos relevantes basados en la consulta
    retriever = docsearch.similarity_search(query, k=1)
    retrieved_context = retriever[0].page_content if retriever else "No relevant context found."

    # Definir el RAG chain (retrieval-augmented generation)
    rag_chain = (
        {  # Ensure context and input are passed correctly
            "context": RunnablePassthrough(),  # Make "context" a passthrough Runnable
            "input": RunnablePassthrough(),    # Make "input" a passthrough Runnable
        }
        | custom_prompt                         # Pipe the prompt with context and input
        | llm                                   # Run the LLM on the custom prompt
        | output_parser                         # Parse the LLM output to a string
    )

    # Ejecutar la consulta y obtener el resultado
    result = rag_chain.invoke({
        "context": retrieved_context,  # Provide the actual context retrieved
        "input": query                 # Provide the actual input (query)
    })
    return result

# Ejemplo de uso
if __name__ == "__main__":
    query = "¿Who is Zara?"
    chat_history = []  # Histórico de chats, si hay
    response = run_llm(query, chat_history)
    print(response)


"""
# Crear el RAG Chain
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA(llm=llm, retriever=retriever)

# Realizar una consulta al sistema RAG
query = "¿Qué información tiene sobre la flor mágica?"
response = qa_chain.run(query)

# Imprimir la respuesta generada
print(response)

info = vectorstore.similarity_search("magical_flower", k=2)
#retriever = vectorstore.as_retriever()

print(info)
"""