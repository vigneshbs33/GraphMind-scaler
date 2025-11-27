"""Global configuration for GraphMind."""

from __future__ import annotations

from pathlib import Path
from typing import Set

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with sensible defaults."""

    # Project paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = DATA_DIR / "uploads"
    CHROMA_DIR: Path = DATA_DIR / "chroma"

    # Model settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    # LLM settings
    LLM_PROVIDER: str = "mock"
    CLAUDE_API_KEY: str = ""
    OLLAMA_MODEL: str = "llama3.2:3b"

    # Search settings
    DEFAULT_TOP_K: int = 5
    HYBRID_ALPHA: float = 0.6

    # File upload limits
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: Set[str] = Field(
        default_factory=lambda: {".txt", ".pdf", ".xml", ".json", ".csv", ".md"}
    )

    # Performance
    CHUNK_SIZE: int = 500
    BATCH_SIZE: int = 10

    class Config:
        env_file = ".env"


settings = Settings()
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.CHROMA_DIR.mkdir(parents=True, exist_ok=True)

