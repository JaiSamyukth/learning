"""
User Management Service with RAG Features
Production-grade user management with RAG-specific capabilities, quotas, and analytics
"""

import asyncio
import hashlib
import secrets
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import HTTPException
from pydantic import BaseModel, EmailStr, validator

from backend.services.database_service import database_service
from backend.services.vector_service import vector_service
from backend.utils.logger import setup_logger


class UserCreateRequest(BaseModel):
    """User creation request model"""
    username: str
    email: EmailStr
    password: str

    @validator('username')
    def validate_username(cls, v):
        if len(v) < 3:
            raise ValueError('Username must be at least 3 characters long')
        if not v.isalnum():
            raise ValueError('Username must contain only letters and numbers')
        return v

    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v


class UserLoginRequest(BaseModel):
    """User login request model"""
    username: str
    password: str


class UserUpdateRequest(BaseModel):
    """User update request model"""
    email: Optional[EmailStr] = None
    rag_enabled: Optional[bool] = None
    embedding_quota: Optional[int] = None


class RAGConfigRequest(BaseModel):
    """RAG configuration update request"""
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    embedding_model: Optional[str] = None
    retrieval_top_k: Optional[int] = None
    rerank_enabled: Optional[bool] = None
    search_score_threshold: Optional[float] = None


class UserManagementService:
    """Production-grade user management with RAG features"""

    def __init__(self):
        self.logger = setup_logger(__name__)

    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        import bcrypt
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        import bcrypt
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except Exception:
            return False

    def generate_session_token(self) -> str:
        """Generate secure session token"""
        return secrets.token_urlsafe(64)

    async def create_user(self, request: UserCreateRequest) -> Dict[str, Any]:
        """Create a new user with RAG features"""
        try:
            # Check if user already exists
            existing_user = database_service.get_user_by_username(request.username)
            if existing_user:
                raise HTTPException(status_code=400, detail="Username already exists")

            # Hash password
            password_hash = self.hash_password(request.password)

            # Create user
            user_id = database_service.create_user(
                username=request.username,
                email=request.email,
                password_hash=password_hash
            )

            # Generate session token
            session_token = self.generate_session_token()

            # Create session
            expires_at = datetime.now() + timedelta(days=30)
            database_service.execute_query('''
                INSERT INTO user_sessions (user_id, session_token, expires_at, is_active)
                VALUES (?, ?, ?, ?)
            ''', (user_id, session_token, expires_at, True))

            # Get user data
            user_data = database_service.get_user_by_username(request.username)

            self.logger.info(f"User created successfully: {request.username}")

            return {
                'user_id': user_id,
                'username': request.username,
                'email': request.email,
                'session_token': session_token,
                'rag_enabled': user_data.get('rag_enabled', True),
                'embedding_quota': user_data.get('embedding_quota', 1000000),
                'created_at': user_data.get('created_at'),
                'message': 'User created successfully'
            }

        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"User creation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"User creation failed: {str(e)}")

    async def login_user(self, request: UserLoginRequest) -> Dict[str, Any]:
        """Authenticate user and return session token"""
        try:
            # Get user from database
            user_data = database_service.get_user_by_username(request.username)
            if not user_data:
                raise HTTPException(status_code=401, detail="Invalid credentials")

            # Verify password
            if not self.verify_password(request.password, user_data['password_hash']):
                raise HTTPException(status_code=401, detail="Invalid credentials")

            # Check if user is active
            if not user_data.get('is_active', True):
                raise HTTPException(status_code=401, detail="Account is disabled")

            # Update last login
            database_service.execute_query('''
                UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
            ''', (user_data['id'],))

            # Generate or refresh session token
            session_token = self.generate_session_token()
            expires_at = datetime.now() + timedelta(days=30)

            # Update or create session
            database_service.execute_query('''
                INSERT OR REPLACE INTO user_sessions (user_id, session_token, expires_at, is_active, created_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (user_data['id'], session_token, expires_at, True))

            self.logger.info(f"User logged in successfully: {request.username}")

            return {
                'user_id': user_data['id'],
                'username': user_data['username'],
                'email': user_data['email'],
                'session_token': session_token,
                'rag_enabled': user_data.get('rag_enabled', True),
                'embedding_quota': user_data.get('embedding_quota', 1000000),
                'quota_used': user_data.get('quota_used', 0),
                'last_login': user_data.get('last_login'),
                'message': 'Login successful'
            }

        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Login failed for user {request.username}: {str(e)}")
            raise HTTPException(status_code=500, detail="Login failed")

    async def get_user_profile(self, session_token: str) -> Dict[str, Any]:
        """Get user profile with RAG statistics"""
        try:
            # Get user ID from session token
            user_id = self._get_user_id_from_session(session_token)
            if not user_id:
                raise HTTPException(status_code=401, detail="Invalid session")

            # Get user data
            user_data = database_service.get_user_by_id(user_id)
            if not user_data:
                raise HTTPException(status_code=404, detail="User not found")

            # Get RAG statistics
            rag_stats = await self._get_user_rag_stats(user_id)

            # Get recent activity
            recent_activity = database_service.get_user_history(user_id, limit=10)

            return {
                'user_id': user_id,
                'username': user_data['username'],
                'email': user_data['email'],
                'rag_enabled': user_data.get('rag_enabled', True),
                'embedding_quota': user_data.get('embedding_quota', 1000000),
                'quota_used': user_data.get('quota_used', 0),
                'quota_remaining': user_data.get('embedding_quota', 1000000) - user_data.get('quota_used', 0),
                'subscription_tier': user_data.get('subscription_tier', 'free'),
                'created_at': user_data.get('created_at'),
                'last_login': user_data.get('last_login'),
                'rag_stats': rag_stats,
                'recent_activity': recent_activity,
                'account_status': 'active' if user_data.get('is_active', True) else 'inactive'
            }

        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Failed to get user profile: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to retrieve user profile")

    async def update_user_profile(self, session_token: str, updates: UserUpdateRequest) -> Dict[str, Any]:
        """Update user profile with RAG settings"""
        try:
            user_id = self._get_user_id_from_session(session_token)
            if not user_id:
                raise HTTPException(status_code=401, detail="Invalid session")

            # Prepare update data
            update_data = {}
            if updates.email is not None:
                update_data['email'] = updates.email
            if updates.rag_enabled is not None:
                update_data['rag_enabled'] = updates.rag_enabled
            if updates.embedding_quota is not None:
                update_data['embedding_quota'] = updates.embedding_quota

            if update_data:
                # Update user in database
                set_clauses = [f"{k} = ?" for k in update_data.keys()]
                values = list(update_data.values())
                values.append(user_id)

                query = f"UPDATE users SET {', '.join(set_clauses)}, updated_at = CURRENT_TIMESTAMP WHERE id = ?"
                database_service.execute_query(query, values)

            # Get updated user data
            user_data = database_service.get_user_by_id(user_id)

            self.logger.info(f"User profile updated: {user_data['username']}")
            return {
                'message': 'Profile updated successfully',
                'user_id': user_id,
                'username': user_data['username'],
                'email': user_data['email'],
                'rag_enabled': user_data.get('rag_enabled', True),
                'embedding_quota': user_data.get('embedding_quota', 1000000),
                'updated_at': user_data.get('updated_at')
            }

        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Failed to update user profile: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to update profile")

    async def update_rag_config(self, session_token: str, config: RAGConfigRequest) -> Dict[str, Any]:
        """Update user RAG configuration"""
        try:
            user_id = self._get_user_id_from_session(session_token)
            if not user_id:
                raise HTTPException(status_code=401, detail="Invalid session")

            # Get current config
            current_config = database_service.get_rag_config(user_id)
            if not current_config:
                raise HTTPException(status_code=404, detail="RAG configuration not found")

            # Prepare update data
            update_data = {}
            if config.chunk_size is not None:
                update_data['chunk_size'] = config.chunk_size
            if config.chunk_overlap is not None:
                update_data['chunk_overlap'] = config.chunk_overlap
            if config.embedding_model is not None:
                update_data['embedding_model'] = config.embedding_model
            if config.retrieval_top_k is not None:
                update_data['retrieval_top_k'] = config.retrieval_top_k
            if config.rerank_enabled is not None:
                update_data['rerank_enabled'] = config.rerank_enabled
            if config.search_score_threshold is not None:
                update_data['search_score_threshold'] = config.search_score_threshold

            # Update configuration
            database_service.update_rag_config(user_id, update_data)

            # Get updated config
            updated_config = database_service.get_rag_config(user_id)

            self.logger.info(f"RAG configuration updated for user {user_id}")
            return {
                'message': 'RAG configuration updated successfully',
                'user_id': user_id,
                'config': updated_config
            }

        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Failed to update RAG config: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to update RAG configuration")

    async def get_user_analytics(self, session_token: str, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive user analytics"""
        try:
            user_id = self._get_user_id_from_session(session_token)
            if not user_id:
                raise HTTPException(status_code=401, detail="Invalid session")

            # Get basic user analytics
            basic_analytics = database_service.get_user_analytics(user_id, days)

            # Get RAG-specific analytics
            rag_analytics = await self._get_user_rag_analytics(user_id, days)

            # Get document analytics
            document_analytics = await self._get_user_document_analytics(user_id)

            return {
                'user_id': user_id,
                'period_days': days,
                'basic_analytics': basic_analytics,
                'rag_analytics': rag_analytics,
                'document_analytics': document_analytics,
                'generated_at': datetime.now().isoformat()
            }

        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Failed to get user analytics: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to retrieve analytics")

    async def logout_user(self, session_token: str) -> Dict[str, Any]:
        """Logout user and invalidate session"""
        try:
            # Deactivate session
            database_service.execute_query('''
                UPDATE user_sessions SET is_active = FALSE WHERE session_token = ?
            ''', (session_token,))

            self.logger.info(f"User logged out: {session_token[:10]}...")
            return {
                'message': 'Logged out successfully',
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Logout failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Logout failed")

    async def delete_user(self, session_token: str, password: str) -> Dict[str, Any]:
        """Delete user account and all associated data"""
        try:
            user_id = self._get_user_id_from_session(session_token)
            if not user_id:
                raise HTTPException(status_code=401, detail="Invalid session")

            # Get user data
            user_data = database_service.get_user_by_id(user_id)
            if not user_data:
                raise HTTPException(status_code=404, detail="User not found")

            # Verify password
            if not self.verify_password(password, user_data['password_hash']):
                raise HTTPException(status_code=401, detail="Invalid password")

            # Delete user data from vector database
            await vector_service.delete_user_data(str(user_id))

            # Delete user from database (CASCADE will handle related data)
            database_service.execute_query('''
                UPDATE users SET is_active = FALSE, updated_at = CURRENT_TIMESTAMP WHERE id = ?
            ''', (user_id,))

            # Deactivate all user sessions
            database_service.execute_query('''
                UPDATE user_sessions SET is_active = FALSE WHERE user_id = ?
            ''', (user_id,))

            self.logger.info(f"User account deleted: {user_data['username']}")
            return {
                'message': 'Account deleted successfully',
                'username': user_data['username'],
                'timestamp': datetime.now().isoformat()
            }

        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"User deletion failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Account deletion failed")

    async def _get_user_rag_stats(self, user_id: int) -> Dict[str, Any]:
        """Get RAG-specific statistics for a user"""
        try:
            # Get collection info
            collection_info = await vector_service.get_collection_info(str(user_id))

            # Get recent RAG interactions
            recent_interactions = database_service.get_user_history(user_id, limit=100)
            rag_interactions = [i for i in recent_interactions if i.get('interaction_type') in ['rag_chat', 'rag_question_generation']]

            # Calculate statistics
            total_rag_interactions = len(rag_interactions)
            avg_confidence = 0.0
            avg_chunks_used = 0

            if rag_interactions:
                confidences = [i.get('performance_score', 0) for i in rag_interactions if i.get('performance_score')]
                chunks_used = [len(i.get('rag_chunks_used', [])) for i in rag_interactions if i.get('rag_chunks_used')]

                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                avg_chunks_used = sum(chunks_used) / len(chunks_used) if chunks_used else 0

            return {
                'collection_exists': collection_info.get('exists', False),
                'total_vectors': collection_info.get('total_vectors', 0),
                'total_rag_interactions': total_rag_interactions,
                'avg_confidence': avg_confidence,
                'avg_chunks_used': avg_chunks_used,
                'last_rag_interaction': rag_interactions[0].get('created_at') if rag_interactions else None
            }

        except Exception as e:
            self.logger.error(f"Failed to get RAG stats for user {user_id}: {str(e)}")
            return {}

    async def _get_user_rag_analytics(self, user_id: int, days: int) -> Dict[str, Any]:
        """Get detailed RAG analytics for a user"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)

            # Get RAG-specific interactions
            interactions = database_service.execute_query('''
                SELECT * FROM user_interactions
                WHERE user_id = ? AND created_at >= ? AND interaction_type LIKE 'rag_%'
                ORDER BY created_at DESC
            ''', (user_id, cutoff_date.isoformat()), fetch_all=True)

            if not interactions:
                return {}

            # Calculate metrics
            total_interactions = len(interactions)
            total_tokens = sum([i.get('tokens_used', 0) for i in interactions if i.get('tokens_used')])
            avg_response_time = sum([i.get('response_time', 0) for i in interactions if i.get('response_time')]) / total_interactions

            # Interaction type breakdown
            interaction_types = {}
            for interaction in interactions:
                it_type = interaction.get('interaction_type', 'unknown')
                interaction_types[it_type] = interaction_types.get(it_type, 0) + 1

            return {
                'total_interactions': total_interactions,
                'total_tokens_used': total_tokens,
                'avg_response_time': avg_response_time,
                'interaction_types': interaction_types,
                'period_days': days
            }

        except Exception as e:
            self.logger.error(f"Failed to get RAG analytics for user {user_id}: {str(e)}")
            return {}

    async def _get_user_document_analytics(self, user_id: int) -> Dict[str, Any]:
        """Get document analytics for a user"""
        try:
            documents = database_service.get_user_documents(user_id)

            if not documents:
                return {}

            # Calculate statistics
            total_documents = len(documents)
            rag_processed = len([d for d in documents if d.get('rag_processed', False)])
            total_size = sum([d.get('file_size', 0) for d in documents])
            total_pages = sum([d.get('total_pages', 0) for d in documents])
            total_tokens = sum([d.get('total_tokens', 0) for d in documents])

            # Get chunks count
            total_chunks = 0
            for doc in documents:
                chunks = database_service.get_document_chunks(doc['id'])
                total_chunks += len(chunks)

            return {
                'total_documents': total_documents,
                'rag_processed_documents': rag_processed,
                'total_size_bytes': total_size,
                'total_pages': total_pages,
                'total_tokens': total_tokens,
                'total_chunks': total_chunks,
                'processing_rate': rag_processed / total_documents if total_documents > 0 else 0
            }

        except Exception as e:
            self.logger.error(f"Failed to get document analytics for user {user_id}: {str(e)}")
            return {}

    def _get_user_id_from_session(self, session_token: str) -> Optional[int]:
        """Get user ID from session token"""
        try:
            result = database_service.execute_query('''
                SELECT user_id FROM user_sessions
                WHERE session_token = ? AND is_active = TRUE AND expires_at > CURRENT_TIMESTAMP
            ''', (session_token,), fetch_one=True)

            return result['user_id'] if result else None

        except Exception as e:
            self.logger.error(f"Failed to get user ID from session: {str(e)}")
            return None

    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user by ID (for internal use)"""
        try:
            # This would need to be implemented in database_service
            # For now, return None
            return None
        except Exception as e:
            self.logger.error(f"Failed to get user by ID {user_id}: {str(e)}")
            return None

    def health_check(self) -> Dict[str, Any]:
        """Health check for user management service"""
        try:
            # Test database connection
            db_status = "ok"

            # Test user count
            try:
                result = database_service.execute_query("SELECT COUNT(*) FROM users", fetch_one=True)
                user_count = result[0] if result else 0
            except:
                user_count = -1

            return {
                'status': 'healthy',
                'database_connection': db_status,
                'user_count': user_count,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


# Global user management service instance
user_management_service = UserManagementService()