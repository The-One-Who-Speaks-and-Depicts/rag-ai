# app/config.py
import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv
from functools import lru_cache

@dataclass(frozen=True)
class ChromaConfig:
    """Configuration for ChromaDB."""
    persistence_path: str = "./persistence"
    collection_name: str = "documents"

@dataclass(frozen=True)
class DeepSeekConfig:
    """Configuration for DeepSeek API."""
    api_key: str
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    temperature: float = 0.1

@dataclass(frozen=True)
class RAGConfig:
    """Configuration for RAG system."""
    context_chunks: int = 3
    system_prompt: str = "You are a helpful assistant that provides accurate answers based on the given context."

@dataclass(frozen=True)
class AppConfig:
    """Main application configuration."""
    chroma: ChromaConfig
    deepseek: DeepSeekConfig
    rag: RAGConfig
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create config instance from environment variables."""
        # Load environment variables
        env_path = Path(__file__).parent.parent / '.env'
        load_dotenv(env_path)
        
        # Validate required environment variables
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        collection_name = os.getenv("COLLECTION_NAME")
        persistence_path = os.getenv("PERSISTENCE_DIR")
        
        if not deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable is required")
        if not collection_name:
            raise ValueError("COLLECTION_NAME environment variable is required")
        if not persistence_path:
            raise ValueError("PERSISTENCE_DIR environment variable is required")
        
        # Create configuration instances
        chroma_config = ChromaConfig(
            collection_name=collection_name,
            persistence_path=persistence_path
        )
        
        deepseek_config = DeepSeekConfig(
            api_key=deepseek_api_key
        )
        
        rag_config = RAGConfig()
        
        return cls(
            chroma=chroma_config,
            deepseek=deepseek_config,
            rag=rag_config
        )
    
    def validate(self) -> None:
        """Validate the configuration."""
        if not self.deepseek.api_key:
            raise ValueError("DeepSeek API key cannot be empty")
        if not self.chroma.collection_name:
            raise ValueError("Collection name cannot be empty")
        if not self.chroma.persistence_path:
            raise ValueError("Persistence path vallowed cannot be empty")
        if self.rag.context_chunks <= 0:
            raise ValueError("Context chunks must be positive")

@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """
    Get the application configuration.
    
    Returns:
        AppConfig: The application configuration instance
        
    Raises:
        ValueError: If configuration validation fails
    """
    config = AppConfig.from_env()
    config.validate()
    print("✅ Configuration loaded and validated successfully")
    return config

def reload_config() -> AppConfig:
    """
    Reload the configuration from environment variables.
    
    Useful for development or when environment variables change.
    """
    get_config.cache_clear()  # Clear the cache
    return get_config()