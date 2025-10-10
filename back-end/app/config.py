# app/config.py
import os
import logging
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv
from functools import lru_cache

# Configure module-specific logger
logger = logging.getLogger(__name__)

# TODO: verbosify this documentation + postinit of DeepSeek config
# Optional: Add custom log formatting for configuration events
class ConfigFormatter(logging.Formatter):
    """Custom formatter for configuration-related log messages."""
    
    def format(self, record):
        if hasattr(record, 'config_component'):
            record.msg = f"[CONFIG:{record.config_component}] {record.msg}"
        return super().format(record)

# Apply custom formatting to the logger
handler = logging.StreamHandler()
handler.setFormatter(ConfigFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

@dataclass(frozen=True)
class ChromaConfig:
    """
    Configuration settings for ChromaDB vector database operations.
    
    This class defines all parameters required to connect to and interact
    with ChromaDB, the vector database used for storing and retrieving
    document embeddings in the RAG system.
    
    Attributes:
        persistence_path (str): 
            The file system path where ChromaDB will persist its data.
            This includes vector indexes, metadata, and collection data.
            The directory will be created automatically if it doesn't exist.
            Default: "./persistence"
            
        collection_name (str): 
            The name of the specific ChromaDB collection containing your
            document embeddings. This collection must be created and 
            populated with embeddings before the RAG system can use it.
            Default: "documents"
            
    Raises:
        ValueError: If any required parameters are missing or invalid when
            validated by the parent AppConfig class.
            
    See Also:
        AppConfig : The parent configuration class that uses this config.
        chromadb.PersistentClient : The ChromaDB client that uses these settings.
        
    Note:
        This is an immutable dataclass. Once created, instances cannot be
        modified. Create new instances if different configuration is needed.
    """   
    persistence_path: str = "./persistence"
    collection_name: str = "documents"

    def __post_init__(self):
        """Log configuration initialization for debugging purposes."""
        logger.info(
            f"ChromaDB configured: collection='{self.collection_name}', "
            f"path='{self.persistence_path}'",
            extra={'config_component': 'chroma'}
        )

@dataclass(frozen=True)
class DeepSeekConfig:
    """
    Configuration settings for the DeepSeek API integration.
    
    This class encapsulates all parameters required to authenticate and 
    interact with the DeepSeek API service, which provides the large 
    language model capabilities for the RAG system's response generation.
    
    Attributes:
        api_key (str): 
            The authentication key required to access the DeepSeek API.
            This is a mandatory security credential that must be obtained
            from the DeepSeek platform and provided via environment variables.
            There is no default value - this must be explicitly provided.
            
    Raises:
        ValueError: If the api_key is missing, empty, or invalid when
            validated by the parent AppConfig class. The api_key is a
            required parameter with no default value.
            
    Security Note:
        The api_key is a sensitive credential that should never be hard-coded
        in source code. It must be provided via environment variables or
        secure secret management systems. The frozen nature of this dataclass
        helps prevent accidental exposure or modification of this sensitive
        information at runtime.
        
    See Also:
        AppConfig : The parent configuration class that instantiates this config.
        openai.OpenAI : The client library used to interact with DeepSeek API.
        
    Note:
        This is an immutable dataclass. Configuration cannot be modified after
        instantiation, ensuring consistent API behavior throughout the
        application lifecycle.
    """
    api_key: str

@dataclass(frozen=True)
class AppConfig:
    """Main application configuration."""
    chroma: ChromaConfig
    deepseek: DeepSeekConfig
    
    @classmethod
    def create_config_from_env(cls) -> "AppConfig":
        """Create config instance from environment variables."""
        logger.info(
            "Starting configuration loading from environment variables",
            extra={'config_component': 'app'}
        )
        # Load environment
        env_path = Path(__file__).parent.parent / '.env'
        load_dotenv(env_path)
        
        # Get required variables
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        collection_name = os.getenv("COLLECTION_NAME")
        persistence_path = os.getenv("PERSISTENCE_DIR")
        
        # Check, whether the variables are not None
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
        
        return cls(
            chroma=chroma_config,
            deepseek=deepseek_config
        )

@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """
    Retrieves the singleton application configuration instance with caching.
    
    This function serves as the primary access point for obtaining the 
    application's configuration throughout the system. It implements a 
    singleton pattern using LRU (Least Recently Used) caching to ensure 
    that configuration is loaded only once per application lifecycle, 
    thereby optimizing performance and ensuring consistency across all 
    components that depend on configuration settings.
    
    The function performs the following operations:
    1. Checks if a configuration instance already exists in the cache
    2. If not cached, creates a new configuration instance by loading
       settings from environment variables and validating them
    3. Returns the cached or newly created configuration instance
    4. Provides console feedback upon successful configuration loading
    
    Returns:
        AppConfig: 
            A fully initialized and validated application configuration 
            instance containing all necessary settings for ChromaDB, 
            DeepSeek API, and RAG system operations. The same instance 
            is returned for all subsequent calls within the same 
            application session due to caching.
            
    Raises:
        ValueError: 
            If any required environment variables are missing, empty, 
            or contain invalid values that prevent successful configuration 
            validation. This includes but is not limited to:
            - Missing DEEPSEEK_API_KEY environment variable
            - Missing COLLECTION_NAME environment variable
            
        chromadb.errors.ChromaError:
            If the specified ChromaDB collection cannot be accessed or 
            does not exist at the configured persistence path.
            
        openai.OpenAIError:
            If the DeepSeek API configuration contains invalid credentials
            or endpoints that cannot be authenticated.
        
    Cache Behavior:
        The function employs an LRU cache with maximum size of 1, 
        meaning only the most recent configuration instance is stored.
        The cache persists until the Python process terminates or until
        explicitly cleared using the `reload_config()` function.
        
    Thread Safety:
        The LRU cache implementation is thread-safe, making this function
        suitable for use in multi-threaded environments such as web 
        servers. Multiple threads will receive the same configuration
        instance without race conditions.
        
    See Also:
        reload_config : Clears the cache and forces configuration reload
        AppConfig.create_config_from_env : Factory method for configuration creation
        AppConfig.validate : Validation logic for configuration integrity
    """
    logger.debug(
        "Configuration access requested - checking cache",
        extra={'config_component': 'cache'}
    )
    
    config = AppConfig.create_config_from_env()
    config.validate()
    
    logger.info(
        "✅ Configuration loaded and validated successfully",
        extra={'config_component': 'app'}
    )
    
    # Log configuration summary for debugging
    logger.debug(
        f"Configuration summary - "
        f"Collection: {config.chroma.collection_name}, ",
        extra={'config_component': 'summary'}
    )
    return config

def reload_config() -> AppConfig:
    """
    Forces a complete reload of the application configuration from environment variables.
    
    This function invalidates the cached configuration instance and triggers
    a fresh loading of all configuration parameters from the environment.
    It is particularly valuable during development, testing, and deployment
    scenarios where configuration changes need to be reflected in the running
    application without requiring a full process restart.
    
    The function performs the following operations in sequence:
    1. Invalidates the LRU cache associated with the `get_config()` function,
       ensuring that any previously cached configuration instance is discarded
    2. Invokes `get_config()` which now bypasses the cache and executes the
       complete configuration loading pipeline from scratch
    3. Returns the newly created configuration instance with updated values
       reflecting current environment variable state
    
    Returns:
        AppConfig: 
            A freshly instantiated application configuration object containing
            the most current values from environment variables. This instance
            undergoes full validation and replaces the previous cached instance
            for all subsequent calls to `get_config()`.
            
    Raises:
        ValueError: 
            If the reloaded configuration fails validation due to missing,
            malformed, or invalid environment variables. The specific error
            messages will indicate which configuration elements require
            attention. Common issues include:
            - DEEPSEEK_API_KEY being unset or revoked
            - COLLECTION_NAME referring to a non-existent ChromaDB collection
            - Invalid numerical values for temperature or context chunks
            
        EnvironmentError:
            If the .env file cannot be read or parsed, or if required
            environment variables are not accessible to the process.
        
    See Also:
        get_config : The primary configuration accessor with caching
        AppConfig.create_config_from_env : The factory method for configuration creation
        AppConfig.validate : Validation logic executed during reload
    """
    logger.info(
        "Configuration reload requested - clearing cache and reloading",
        extra={'config_component': 'cache'}
    )
    
    cache_info_before = get_config.cache_info()
    get_config.cache_clear()
    
    logger.debug(
        f"Cache cleared - previous stats: {cache_info_before}",
        extra={'config_component': 'cache'}
    )
    
    try:
        config = get_config()
        logger.info(
            "Configuration reload completed successfully",
            extra={'config_component': 'app'}
        )
        return config
    except Exception as e:
        logger.critical(
            f"Configuration reload failed: {str(e)}",
            extra={'config_component': 'app'},
            exc_info=True
        )
        raise