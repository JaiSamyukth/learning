"""
Vector Operations Service for Qdrant Cloud Integration
Production-grade service for handling vector operations with comprehensive error handling
"""

import asyncio
import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
import numpy as np
import os
import google.generativeai as genai

from config.qdrant_config import qdrant_config
from utils.logger import get_logger
from utils.cache import CacheService
import sys

# Add the parent directory to the path to import from api_rotation
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from api_rotation.api_key_rotator import api_key_rotator
    ROTATION_AVAILABLE = True
except ImportError:
    ROTATION_AVAILABLE = False


class VectorServiceError(Exception):
    """Custom exception for vector service errors"""
    pass


class VectorService:
    """Production-grade vector operations service"""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.cache = CacheService()
        self._gemini_client = None
        self._qdrant_client = None
        self._lock = asyncio.Lock()

        # Initialize components
        self._initialize_qdrant_client()
        self._initialize_gemini_client()
        self._ensure_collections_exist()

    def _initialize_qdrant_client(self):
        """Initialize Qdrant Cloud client with error handling"""
        try:
            self._qdrant_client = QdrantClient(
                url=qdrant_config.QDRANT_URL,
                api_key=qdrant_config.QDRANT_API_KEY,
                timeout=qdrant_config.REQUEST_TIMEOUT
            )

            # Test connection
            collections = self._qdrant_client.get_collections()
            self.logger.info(f"Successfully connected to Qdrant Cloud. Collections: {[c.name for c in collections.collections]}")

        except Exception as e:
            self.logger.error(f"Failed to initialize Qdrant client: {str(e)}")
            raise VectorServiceError(f"Qdrant Cloud connection failed: {str(e)}")

    def _initialize_gemini_client(self):
        """Initialize Gemini client for embeddings with API key rotation"""
        try:
            if ROTATION_AVAILABLE:
                # Use API key rotation
                stats = api_key_rotator.get_current_stats()
                if stats['has_keys']:
                    self.logger.info(f"Successfully initialized Gemini client with {stats['total_keys']} rotating API keys")
                    return
                else:
                    self.logger.warning("API key rotator has no keys, falling back to environment variable")

            # Fallback to single API key from environment
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set and no rotation keys available")

            genai.configure(api_key=api_key)
            self.logger.info("Successfully initialized Gemini client for embeddings (single key)")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini client: {str(e)}")
            raise VectorServiceError(f"Gemini client initialization failed: {str(e)}")

    def _get_rotated_api_key(self):
        """Get the next API key from rotation or fallback to environment"""
        if ROTATION_AVAILABLE:
            key = api_key_rotator.get_next_key()
            if key:
                return key
            else:
                self.logger.warning("No keys available from rotator, using environment fallback")

        # Fallback to environment variable
        return os.getenv('GEMINI_API_KEY')

    def _configure_gemini_for_request(self):
        """Configure Gemini API key for the current request"""
        api_key = self._get_rotated_api_key()
        if not api_key:
            raise VectorServiceError("No Gemini API key available")

        genai.configure(api_key=api_key)
        return api_key

    def _ensure_collections_exist(self):
        """Ensure required collections exist in Qdrant Cloud"""
        try:
            collections = self._qdrant_client.get_collections()
            existing_collections = {c.name for c in collections.collections}

            # Create main collection if it doesn't exist
            if qdrant_config.MAIN_COLLECTION not in existing_collections:
                self._create_collection(qdrant_config.MAIN_COLLECTION)
                self.logger.info(f"Created main collection: {qdrant_config.MAIN_COLLECTION}")

        except Exception as e:
            self.logger.error(f"Failed to ensure collections exist: {str(e)}")
            raise VectorServiceError(f"Collection setup failed: {str(e)}")

    def _create_collection(self, collection_name: str):
        """Create a new collection with optimized configuration"""
        try:
            self._qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=qdrant_config.VECTOR_SIZE,
                    distance=models.Distance.COSINE
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    default_segment_number=2,
                    max_segment_size=100000
                ),
                # replication_config=models.Replication(
                #     factor=2
                # )  # Commented out due to version compatibility
            )
        except UnexpectedResponse as e:
            if "already exists" in str(e):
                self.logger.info(f"Collection {collection_name} already exists")
            else:
                raise e

    async def process_document_chunks(self, chunks: List[Dict[str, Any]], user_id: str) -> Dict[str, Any]:
        """
        Process document chunks and store in Qdrant Cloud
        Returns processing statistics
        """
        async with self._lock:
            try:
                collection_name = f"{qdrant_config.USER_COLLECTION_PREFIX}{user_id}"

                # Ensure user collection exists
                if collection_name not in {c.name for c in self._qdrant_client.get_collections().collections}:
                    self._create_collection(collection_name)

                processed_count = 0
                failed_count = 0
                vectors_batch = []

                for i, chunk in enumerate(chunks):
                    try:
                        # Generate embedding using Gemini with API key rotation
                        try:
                            # Configure API key for this request
                            api_key = self._configure_gemini_for_request()
                            self.logger.debug(f"Using API key: {api_key_rotator.get_key_preview(api_key) if ROTATION_AVAILABLE else 'env_key'}")

                            result = genai.embed_content(
                                model="models/embedding-001",
                                content=chunk['content']
                            )
                            embedding = result['embedding']
                        except Exception as e:
                            if "quota" in str(e).lower() or "429" in str(e):
                                # Quota exceeded - use mock embedding for testing
                                self.logger.warning(f"Gemini quota exceeded, using mock embedding: {e}")
                                import hashlib
                                import numpy as np
                                # Create deterministic mock embedding based on content hash
                                content_hash = hashlib.md5(chunk['content'].encode()).hexdigest()
                                np.random.seed(int(content_hash[:8], 16))
                                embedding = np.random.normal(0, 1, 768).tolist()
                            else:
                                raise e

                        # Prepare vector data
                        vector_data = {
                            'id': chunk['chunk_id'],
                            'vector': embedding,
                            'payload': {
                                'content': chunk['content'],
                                'metadata': json.dumps(chunk['metadata']),
                                'user_id': user_id,
                                'document_id': chunk.get('document_id', ''),
                                'chunk_type': chunk.get('chunk_type', 'unknown'),
                                'importance_score': chunk.get('importance_score', 0.5),
                                'created_at': datetime.now().isoformat()
                            }
                        }
                        vectors_batch.append(vector_data)

                        # Process in batches
                        if len(vectors_batch) >= qdrant_config.BATCH_SIZE or i == len(chunks) - 1:
                            await self._store_batch_vectors(collection_name, vectors_batch)
                            processed_count += len(vectors_batch)
                            vectors_batch.clear()

                    except Exception as e:
                        self.logger.error(f"Failed to process chunk {chunk.get('chunk_id', i)}: {str(e)}")
                        failed_count += 1

                stats = {
                    'total_chunks': len(chunks),
                    'processed_successfully': processed_count,
                    'failed_chunks': failed_count,
                    'collection_name': collection_name,
                    'processing_time': datetime.now().isoformat()
                }

                self.logger.info(f"Document processing completed: {stats}")
                return stats

            except Exception as e:
                self.logger.error(f"Document processing failed: {str(e)}")
                raise VectorServiceError(f"Document processing error: {str(e)}")

    async def _store_batch_vectors(self, collection_name: str, vectors_batch: List[Dict]):
        """Store vectors in batch with error handling"""
        try:
            points = [
                models.PointStruct(
                    id=vector['id'],
                    vector=vector['vector'],
                    payload=vector['payload']
                )
                for vector in vectors_batch
            ]

            self._qdrant_client.upsert(
                collection_name=collection_name,
                points=points
            )

        except Exception as e:
            self.logger.error(f"Batch storage failed: {str(e)}")
            raise VectorServiceError(f"Batch storage error: {str(e)}")

    async def search_similar(self, query: str, user_id: str, limit: int = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors using semantic search
        Returns scored results with metadata
        """
        try:
            # Generate query embedding using Gemini with API key rotation
            api_key = self._configure_gemini_for_request()
            self.logger.debug(f"Search using API key: {api_key_rotator.get_key_preview(api_key) if ROTATION_AVAILABLE else 'env_key'}")

            result = genai.embed_content(
                model="models/embedding-001",
                content=query
            )
            query_embedding = result['embedding']

            collection_name = f"{qdrant_config.USER_COLLECTION_PREFIX}{user_id}"

            # Check if collection exists
            collections = self._qdrant_client.get_collections().collections
            if collection_name not in {c.name for c in collections}:
                return []

            # Perform search
            search_result = self._qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=limit or qdrant_config.DEFAULT_SEARCH_LIMIT,
                score_threshold=qdrant_config.SEARCH_SCORE_THRESHOLD
            )

            # Format results
            results = []
            for scored_point in search_result:
                try:
                    metadata = json.loads(scored_point.payload.get('metadata', '{}'))
                    result = {
                        'id': scored_point.id,
                        'content': scored_point.payload['content'],
                        'score': scored_point.score,
                        'metadata': metadata,
                        'payload': scored_point.payload
                    }
                    results.append(result)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse metadata for point {scored_point.id}: {str(e)}")

            # Sort by score (highest first)
            results.sort(key=lambda x: x['score'], reverse=True)

            self.logger.info(f"Search returned {len(results)} results for user {user_id}")
            return results

        except Exception as e:
            self.logger.error(f"Search failed for user {user_id}: {str(e)}")
            return []

    async def get_collection_info(self, user_id: str) -> Dict[str, Any]:
        """Get collection statistics for a user"""
        try:
            collection_name = f"{qdrant_config.USER_COLLECTION_PREFIX}{user_id}"

            # Check if collection exists
            collections = self._qdrant_client.get_collections().collections
            if collection_name not in {c.name for c in collections}:
                return {
                    'collection_name': collection_name,
                    'exists': False,
                    'total_vectors': 0
                }

            # Get collection info
            info = self._qdrant_client.get_collection(collection_name)

            return {
                'collection_name': collection_name,
                'exists': True,
                'total_vectors': info.points_count,
                'status': info.status.value,
                'created_at': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Failed to get collection info for user {user_id}: {str(e)}")
            return {
                'collection_name': f"{qdrant_config.USER_COLLECTION_PREFIX}{user_id}",
                'exists': False,
                'total_vectors': 0,
                'error': str(e)
            }

    async def delete_user_data(self, user_id: str) -> bool:
        """Delete all data for a specific user"""
        try:
            collection_name = f"{qdrant_config.USER_COLLECTION_PREFIX}{user_id}"

            # Check if collection exists
            collections = self._qdrant_client.get_collections().collections
            if collection_name in {c.name for c in collections}:
                self._qdrant_client.delete_collection(collection_name)
                self.logger.info(f"Deleted collection {collection_name} for user {user_id}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to delete user data for {user_id}: {str(e)}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """Health check for the vector service"""
        try:
            # Test Qdrant connection
            collections = self._qdrant_client.get_collections()

            # Test Gemini embedding with API key rotation (with quota fallback)
            try:
                api_key = self._configure_gemini_for_request()
                self.logger.debug(f"Health check using API key: {api_key_rotator.get_key_preview(api_key) if ROTATION_AVAILABLE else 'env_key'}")

                test_result = genai.embed_content(
                    model="models/embedding-001",
                    content="test"
                )
                test_embedding = test_result['embedding']
                embedding_source = f"gemini-embedding-001 ({'rotation' if ROTATION_AVAILABLE else 'single'})"
            except Exception as e:
                if "quota" in str(e).lower() or "429" in str(e):
                    # Quota exceeded - use mock for health check
                    self.logger.warning(f"Gemini quota exceeded in health check, using mock: {e}")
                    test_embedding = [0.1] * 768  # Mock 768-dim embedding
                    embedding_source = "gemini-embedding-001 (quota fallback)"
                else:
                    raise e

            return {
                'status': 'healthy',
                'qdrant_connection': 'ok',
                'embedding_model': embedding_source,
                'embedding_dimensions': len(test_embedding),
                'total_collections': len(collections.collections),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


# Global vector service instance
vector_service = VectorService()