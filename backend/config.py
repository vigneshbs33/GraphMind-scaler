"""Global configuration for GraphMind - Production-ready."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Set

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings with sensible defaults and validation."""

    # Environment
    ENVIRONMENT: str = "development"  # development, staging, production
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # Project paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = DATA_DIR / "uploads"
    CHROMA_DIR: Path = DATA_DIR / "chroma"
    LOG_DIR: Path = BASE_DIR / "logs"

    # Model settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    # LLM settings
    LLM_PROVIDER: str = "gemini"  # Options: "mock", "gemini", "claude", "ollama"
    GEMINI_API_KEY: str = Field(default="AIzaSyASunHPAbRNSxHUucdAfay1V_-Chch9MiQ", description="Gemini API key")
    GEMINI_API_URL: str = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    GEMINI_TIMEOUT: int = 30
    CLAUDE_API_KEY: str = ""
    OLLAMA_MODEL: str = "llama3.2:3b"

    # Search settings
    DEFAULT_TOP_K: int = Field(default=5, ge=1, le=50)
    MAX_TOP_K: int = 50
    HYBRID_ALPHA: float = Field(default=0.6, ge=0.0, le=1.0)

    # File upload limits
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: Set[str] = Field(
        default_factory=lambda: {".txt", ".pdf", ".xml", ".json", ".csv", ".md"}
    )

    # Performance
    CHUNK_SIZE: int = Field(default=500, ge=100, le=2000)
    BATCH_SIZE: int = Field(default=10, ge=1, le=100)
    MAX_CONCURRENT_UPLOADS: int = 5

    # Security
    ALLOWED_ORIGINS: List[str] = Field(
        default_factory=lambda: ["http://localhost:8000", "http://127.0.0.1:8000", "*"]
    )
    RATE_LIMIT_ENABLED: bool = False
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # seconds

    # Graph settings
    MAX_GRAPH_NODES: int = 10000
    MAX_GRAPH_EDGES: int = 50000
    SEMANTIC_SIMILARITY_THRESHOLD: float = Field(default=0.7, ge=0.0, le=1.0)

    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment value."""
        allowed = {"development", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"Log level must be one of {allowed}")
        return v.upper()

    def __init__(self, **kwargs):
        """Initialize settings and create directories."""
        super().__init__(**kwargs)
        
        # Create directories
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        # Set log level
        logging.getLogger().setLevel(getattr(logging, self.LOG_LEVEL))
        
        # Load API key from environment if not set
        if not self.GEMINI_API_KEY or self.GEMINI_API_KEY == "":
            self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
        
        if self.ENVIRONMENT == "production":
            self.DEBUG = False
            if not self.GEMINI_API_KEY:
                logger.warning("GEMINI_API_KEY not set in production environment")
        
        logger.info("Settings initialized: environment=%s, debug=%s", self.ENVIRONMENT, self.DEBUG)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
