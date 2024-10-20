import os
from dotenv import load_dotenv

# Cargar las variables desde el archivo .env
load_dotenv()

class Settings:
    # Variables de entorno
    cohere_key = os.getenv('COHERE_API_KEY')
    #DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///:memory:")
    #DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "False").lower() in ("true", "1", "yes")

# Instancia de configuración global
settings = Settings()