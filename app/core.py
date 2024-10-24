from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_community.utils.math import cosine_similarity
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

# Umbral de similitud
THRESHOLD = 0.8  # Ajustar según las necesidades. Por ejemplo: La misma pregunta en inglés y español suele tener una similitud coseno de 0,7. Por eso un 0.8 evita que traiga una respuesta en otro idioma desde la base de datos de preguntas pasadas.

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

# Definir la función para detectar idioma de la query
def detect_lang(query: str):

    # Inicializar el LLM (ChatOpenAI) para la generación de respuestas
    llm = ChatOpenAI(model_name="gpt-4o", verbose=True, temperature=0)

    # Crear un prompt personalizado con las instrucciones deseadas
    leng_prompt = PromptTemplate(
        input_variables=["input"],
        template=(
            "Based on the language of the following question, respond only with the name of that language: {input}\n"  
            "Provide a one-word response only.\n"
        )
    )

    # Crear un parser para convertir la respuesta del LLM en cadena de texto
    output_parser = StrOutputParser()

    # Definir el RAG chain (retrieval-augmented generation)
    chain = (
        {"input": RunnablePassthrough()}
        | leng_prompt                         
        | llm                                  
        | output_parser                     
    )

    # Ejecutar la consulta y obtener el resultado
    language = chain.invoke({
        "input": query                
    })

    print(language)
    return language


# Definir la función para ejecutar el LLM con RAG utilizando Chroma
def run_llm(query: str):
    # Cargar el vectorstore persistente de Chrom
    docsearch = load_or_create_vectorstore()
    
    # Inicializar el LLM (ChatOpenAI) para la generación de respuestas
    llm = ChatOpenAI(model_name="gpt-4o", verbose=True, temperature=0)

    language = detect_lang(query=query)

    # Crear un prompt personalizado con las instrucciones deseadas
    custom_prompt = PromptTemplate(
        input_variables=["context","input","language"],
        template=(
            "Relevant context: {context}\n"
            "Question: {input}\n\n"
            "Answer the following question while following these rules:\n"
            "- Respond in only one sentence.\n"
            "- Include emojis that summarize the content of the answer.\n"
            "- The response must be in third person only.\n"
            "- The response should be concise and clear.\n"
            "- Include any necessary details while remaining in third person.\n"
            "- Avoid using any first-person pronouns."
            "- Provide the answer in the following language: {language}.\n\n"
            "Answer:"
        )
    )

    # Crear un parser para convertir la respuesta del LLM en cadena de texto
    output_parser = StrOutputParser()

    # Buscar los documentos relevantes basados en la consulta
    retriever = docsearch.similarity_search(query, k=1)
    retrieved_context = retriever[0].page_content if retriever else "No relevant context found."

    # Definir el RAG chain
    rag_chain = (
        RunnablePassthrough()  
        | custom_prompt                         
        | llm                                   
        | output_parser                         
    )

    # Ejecutar la consulta y obtener el resultado
    result = rag_chain.invoke(
        {
            "context": retrieved_context,
            "input": query,
            "language": language
        }
    )
    return result

# Crear la vectorDB para las preguntas
question_store = Chroma(
    collection_name="questions",
    embedding_function=emb,
    persist_directory="./question_db"
)


def get_response(query):

    # Obtiene la pregunta almacenada más similar
    question_retriever = question_store.similarity_search(query, k=1)

    # Crea los embedings de las preguntas y la pregunta almacenada más similar
    query_embedding = emb.embed_query(query)
    question_retriever_embedding = emb.embed_query(question_retriever[0].metadata['question'])

    if question_retriever:

        similarity = cosine_similarity([query_embedding], [question_retriever_embedding])

        if similarity >= THRESHOLD:
            return question_retriever[0].page_content
    
    # Si no hay respuesta en caché, genera una nueva
    new_response = run_llm(query)  
    
    # Almacena la nueva pregunta y su respuesta
    question_store.add_documents([Document(page_content=new_response, metadata={"question": query})])
    
    return new_response
