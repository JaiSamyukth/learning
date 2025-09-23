"""
RAG Integration Service
Bridges the existing PDF service with the RAG system for seamless integration
"""

import asyncio
import os
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from utils.logger import get_logger
from config.settings import settings

# RAG system imports with fallback
try:
    from services.rag_service import rag_service
    from services.vector_service import vector_service
    from services.database_service import database_service
    from services.rag_chat_service import rag_chat_service
    from config.qdrant_config import qdrant_config
    RAG_AVAILABLE = True
except ImportError as e:
    RAG_AVAILABLE = False
    rag_logger = get_logger("rag_integration")
    rag_logger.warning(f"RAG system not available: {e}")

class RAGIntegrationService:
    """Service to integrate existing PDF processing with RAG capabilities"""

    def __init__(self):
        self.RAG_AVAILABLE = RAG_AVAILABLE
        self.rag_enabled = RAG_AVAILABLE
        self.logger = get_logger("rag_integration")

        if RAG_AVAILABLE:
            self.logger.info("RAG integration service initialized successfully")
        else:
            self.logger.warning("RAG integration service running in fallback mode")

    def __init__(self):
        self.logger = get_logger("rag_integration")
        self.rag_enabled = RAG_AVAILABLE and self._check_rag_health()
        
        if self.rag_enabled:
            self.logger.info("RAG integration service initialized successfully")
        else:
            self.logger.warning("RAG integration service running in fallback mode")
    
    def _check_rag_health(self) -> bool:
        """Check if RAG services are healthy"""
        try:
            if not RAG_AVAILABLE:
                return False
            
            # Test vector service connection
            health = vector_service.health_check()
            if health.get('status') != 'healthy':
                self.logger.warning(f"Vector service unhealthy: {health}")
                return False
            
            # Test database service
            db_health = database_service.health_check()
            if db_health.get('status') != 'healthy':
                self.logger.warning(f"Database service unhealthy: {db_health}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"RAG health check failed: {e}")
            return False
    
    async def process_pdf_with_rag(self, file_path: str, filename: str, user_token: str) -> Dict[str, Any]:
        """
        Process a PDF with RAG capabilities if available, otherwise return basic info
        """
        try:
            if not self.rag_enabled:
                return {
                    "rag_processed": False,
                    "message": "RAG processing not available, using basic processing",
                    "fallback": True
                }
            
            # Convert user token to user ID (simple hash-based approach for now)
            user_id = self._get_user_id_from_token(user_token)
            
            # Ensure user exists in RAG database
            await self._ensure_user_exists(user_id, user_token)
            
            # Process document with RAG
            self.logger.info(f"Processing {filename} with RAG for user {user_id}")
            result = await rag_service.process_document(file_path, filename, user_id)
            
            # Update result with integration info
            result.update({
                "rag_processed": True,
                "user_id": user_id,
                "integration_timestamp": datetime.now().isoformat()
            })
            
            self.logger.info(f"RAG processing completed for {filename}")
            return result
            
        except Exception as e:
            self.logger.error(f"RAG processing failed for {filename}: {e}")
            return {
                "rag_processed": False,
                "error": str(e),
                "message": "RAG processing failed, falling back to basic processing",
                "fallback": True
            }
    
    async def get_rag_status_for_user(self, user_token: str) -> Dict[str, Any]:
        """Get RAG status for a user"""
        try:
            if not self.rag_enabled:
                return {
                    "rag_available": False,
                    "message": "RAG system not available"
                }
            
            user_id = self._get_user_id_from_token(user_token)
            
            # Check if user has RAG documents
            collection_info = await vector_service.get_collection_info(user_id)
            documents = database_service.get_user_documents(int(user_id))
            
            return {
                "rag_available": True,
                "user_id": user_id,
                "has_rag_documents": collection_info.get('exists', False),
                "total_vectors": collection_info.get('total_vectors', 0),
                "total_documents": len(documents),
                "rag_documents": len([d for d in documents if d.get('rag_processed', False)])
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get RAG status: {e}")
            return {
                "rag_available": False,
                "error": str(e)
            }
    
    async def enhanced_chat(self, message: str, user_token: str) -> Dict[str, Any]:
        """Enhanced chat with RAG if available"""
        try:
            if not self.rag_enabled:
                return {
                    "rag_used": False,
                    "message": "RAG not available, use regular chat"
                }
            
            # Use RAG chat service
            result = await rag_chat_service.chat(message, user_token, use_rag=True)
            return result
            
        except Exception as e:
            self.logger.error(f"Enhanced chat failed: {e}")
            return {
                "rag_used": False,
                "error": str(e),
                "message": "RAG chat failed, use regular chat"
            }
    
    async def enhanced_question_generation(self, user_token: str, topic: str = None, 
                                         count: int = 25, mode: str = "practice") -> Dict[str, Any]:
        """Enhanced question generation with RAG"""
        try:
            if not self.rag_enabled:
                return {
                    "rag_used": False,
                    "message": "RAG not available, use regular question generation"
                }
            
            result = await rag_chat_service.generate_questions_rag(user_token, topic, count, mode)
            return result
            
        except Exception as e:
            self.logger.error(f"Enhanced question generation failed: {e}")
            return {
                "rag_used": False,
                "error": str(e),
                "message": "RAG question generation failed"
            }
    
    def _get_user_id_from_token(self, user_token: str) -> str:
        """Convert user token to user ID"""
        # For now, use the token directly as user ID
        # In production, this would involve proper token validation
        return user_token.replace("user_", "")
    
    async def _ensure_user_exists(self, user_id: str, user_token: str):
        """Ensure user exists in RAG database"""
        try:
            # Check if user exists
            user_data = database_service.get_user_by_id(int(user_id))
            
            if not user_data:
                # Create user with basic info
                username = f"user_{user_id}"
                email = f"{username}@lumina-iq.local"
                password_hash = "legacy_user"  # Placeholder for legacy users
                
                database_service.create_user(username, email, password_hash)
                self.logger.info(f"Created RAG user for legacy token: {user_token}")
                
        except Exception as e:
            self.logger.warning(f"Could not ensure user exists: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for RAG integration"""
        try:
            return {
                "status": "healthy" if self.rag_enabled else "degraded",
                "rag_available": RAG_AVAILABLE,
                "rag_enabled": self.rag_enabled,
                "services": {
                    "vector_service": vector_service.health_check() if RAG_AVAILABLE else {"status": "unavailable"},
                    "database_service": database_service.health_check() if RAG_AVAILABLE else {"status": "unavailable"}
                } if RAG_AVAILABLE else {"status": "unavailable"},
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# Global instance
rag_integration_service = RAGIntegrationService()
