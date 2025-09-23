"""
Qdrant Cloud Configuration
Production-grade configuration for vector database operations
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class QdrantConfig(BaseSettings):
    """Qdrant Cloud configuration settings"""

    # Qdrant Cloud credentials
    QDRANT_URL: str = Field(
        default="https://1f6b3bbc-d09e-40c2-b333-0a823825f876.europe-west3-0.gcp.cloud.qdrant.io:6333",
        description="Qdrant Cloud URL"
    )
    QDRANT_API_KEY: str = Field(
        default="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.wnnpEmwXoHOjNJ1CTdGaFgqoG7zgLO3O-8bhUbPmK_o",
        description="Qdrant Cloud API Key"
    )

    # Vector configuration
    EMBEDDING_MODEL: str = Field(
        default="gemini-embedding-001",
        description="Gemini embedding model for text encoding"
    )
    VECTOR_SIZE: int = Field(
        default=768,
        description="Dimension of vectors"
    )
    GEMINI_API_KEY: str = Field(
        default="",
        description="Gemini API key for embeddings"
    )

    # Collection configuration
    MAIN_COLLECTION: str = Field(
        default="lumina_iq_documents",
        description="Main collection for document vectors"
    )
    USER_COLLECTION_PREFIX: str = Field(
        default="user_",
        description="Prefix for user-specific collections"
    )

    # Chunking configuration
    DEFAULT_CHUNK_SIZE: int = Field(
        default=1000,
        description="Default chunk size in characters"
    )
    DEFAULT_CHUNK_OVERLAP: int = Field(
        default=200,
        description="Default overlap between chunks"
    )

    # Search configuration
    DEFAULT_SEARCH_LIMIT: int = Field(
        default=5,
        description="Default number of results to return"
    )
    SEARCH_SCORE_THRESHOLD: float = Field(
        default=0.7,
        description="Minimum similarity score for results"
    )

    # Performance configuration
    BATCH_SIZE: int = Field(
        default=100,
        description="Batch size for embedding operations"
    )
    REQUEST_TIMEOUT: int = Field(
        default=30,
        description="Request timeout in seconds"
    )

    # Caching configuration
    CACHE_TTL: int = Field(
        default=3600,
        description="Cache TTL in seconds"
    )

    # Rate limiting
    MAX_REQUESTS_PER_MINUTE: int = Field(
        default=60,
        description="Maximum requests per minute per user"
    )

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global configuration instance
qdrant_config = QdrantConfig()

# Environment variable validation
if not qdrant_config.QDRANT_URL or not qdrant_config.QDRANT_API_KEY:
    raise ValueError("Qdrant Cloud URL and API Key must be configured")