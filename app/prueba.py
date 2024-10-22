from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
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
persist_directory = "./db_index2"

# Definir modelo de embeddings
emb = OpenAIEmbeddings(model="text-embedding-3-small")

if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

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

info = vectorstore.similarity_search("who is zara?", k=2)
print(info)