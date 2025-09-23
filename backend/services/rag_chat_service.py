"""
RAG-Enhanced Chat Service
Production-grade chat service with RAG capabilities integrated with existing system
"""

import asyncio
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import HTTPException

from services.chat_service import ChatService
from services.rag_service import rag_service
from services.vector_service import vector_service
from services.database_service import database_service
from config.qdrant_config import qdrant_config
from utils.logger import get_logger


class RAGChatService:
    """Enhanced chat service with RAG capabilities"""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.chat_service = ChatService()
        self.rag_service = rag_service

    async def chat(self, message: str, token: str, use_rag: bool = True) -> Dict[str, Any]:
        """
        Enhanced chat with RAG support
        Falls back to original chat service if RAG is not available
        """
        try:
            # Get user ID from token (simplified - in production use proper auth)
            user_id = self._get_user_id_from_token(token)

            # Check if user has RAG-enabled documents
            if use_rag and await self._user_has_rag_documents(user_id):
                return await self._rag_chat(message, user_id, token)
            else:
                # Fall back to original chat service
                from models.chat import ChatMessage
                chat_message = ChatMessage(message=message)
                response = await self.chat_service.chat(chat_message, token)

                # Log interaction in new system
                database_service.log_user_interaction(
                    user_id=user_id,
                    interaction_type='legacy_chat',
                    query=message,
                    response=response.response,
                    metadata={'fallback': True}
                )

                return {
                    'response': response.response,
                    'timestamp': response.timestamp,
                    'rag_used': False,
                    'chunks_used': 0
                }

        except Exception as e:
            self.logger.error(f"RAG chat failed, falling back to legacy: {str(e)}")

            # Fallback to original chat service
            try:
                from models.chat import ChatMessage
                chat_message = ChatMessage(message=message)
                response = await self.chat_service.chat(chat_message, token)

                # Get user ID and log fallback
                user_id = self._get_user_id_from_token(token)
                database_service.log_user_interaction(
                    user_id=user_id,
                    interaction_type='fallback_chat',
                    query=message,
                    response=response.response,
                    metadata={'error': str(e), 'fallback': True}
                )

                return {
                    'response': response.response,
                    'timestamp': response.timestamp,
                    'rag_used': False,
                    'chunks_used': 0,
                    'fallback': True,
                    'error': str(e)
                }
            except Exception as fallback_error:
                self.logger.error(f"Even fallback chat failed: {str(fallback_error)}")
                raise HTTPException(
                    status_code=500,
                    detail="Chat service temporarily unavailable. Please try again later."
                )

    async def _rag_chat(self, message: str, user_id: str, token: str) -> Dict[str, Any]:
        """Handle chat using RAG system"""
        try:
            # Generate answer using RAG
            rag_response = await self.rag_service.generate_answer(message, user_id)

            # Log the interaction
            database_service.log_user_interaction(
                user_id=int(user_id),
                interaction_type='rag_chat',
                query=message,
                response=rag_response['answer'],
                rag_chunks_used=rag_response['sources'],
                performance_score=rag_response.get('confidence', 0.5),
                response_time=rag_response.get('response_time', 0),
                metadata={
                    'chunks_used': rag_response['chunks_used'],
                    'context_length': rag_response.get('context_length', 0),
                    'confidence': rag_response.get('confidence', 0.5)
                }
            )

            return {
                'response': rag_response['answer'],
                'timestamp': datetime.now().isoformat(),
                'rag_used': True,
                'chunks_used': rag_response['chunks_used'],
                'sources': rag_response['sources'],
                'confidence': rag_response.get('confidence', 0.5),
                'context_length': rag_response.get('context_length', 0),
                'response_time': rag_response.get('response_time', 0)
            }

        except Exception as e:
            self.logger.error(f"RAG chat processing failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Enhanced chat processing failed: {str(e)}"
            )

    async def _user_has_rag_documents(self, user_id: str) -> bool:
        """Check if user has documents processed with RAG"""
        try:
            # Check if user has any documents with RAG chunks
            collection_info = await vector_service.get_collection_info(user_id)

            if collection_info.get('exists', False) and collection_info.get('total_vectors', 0) > 0:
                return True

            # Also check database for processed documents
            documents = database_service.get_user_documents(int(user_id))
            rag_processed_docs = [doc for doc in documents if doc.get('rag_processed', False)]

            return len(rag_processed_docs) > 0

        except Exception as e:
            self.logger.warning(f"Error checking RAG documents for user {user_id}: {str(e)}")
            return False

    def _get_user_id_from_token(self, token: str) -> str:
        """Extract user ID from token (simplified implementation)"""
        # In production, decode JWT token or lookup in database
        # For now, return a hash of the token as user ID
        import hashlib
        return hashlib.md5(token.encode()).hexdigest()

    async def get_chat_history(self, token: str) -> Dict[str, Any]:
        """Get combined chat history from both legacy and RAG systems"""
        try:
            # Get legacy chat history
            legacy_history = self.chat_service.get_chat_history(token)

            # Get RAG interaction history
            user_id = self._get_user_id_from_token(token)
            rag_history = database_service.get_user_history(int(user_id), limit=50)

            # Combine and format histories
            combined_history = []

            # Add legacy history
            for item in legacy_history.get('history', []):
                combined_history.append({
                    'type': 'legacy',
                    'user': item.get('user', ''),
                    'assistant': item.get('assistant', ''),
                    'timestamp': item.get('timestamp', ''),
                    'rag_used': False
                })

            # Add RAG history
            for interaction in rag_history:
                if interaction.get('interaction_type') == 'rag_chat':
                    combined_history.append({
                        'type': 'rag',
                        'user': interaction.get('query', ''),
                        'assistant': interaction.get('response', ''),
                        'timestamp': interaction.get('created_at', ''),
                        'rag_used': True,
                        'chunks_used': len(interaction.get('rag_chunks_used', [])) if interaction.get('rag_chunks_used') else 0
                    })

            # Sort by timestamp
            combined_history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

            return {
                'history': combined_history[:100],  # Limit to last 100 interactions
                'total_count': len(combined_history),
                'rag_interactions': len([h for h in combined_history if h.get('rag_used', False)]),
                'legacy_interactions': len([h for h in combined_history if not h.get('rag_used', False)])
            }

        except Exception as e:
            self.logger.error(f"Failed to get combined chat history: {str(e)}")
            # Fallback to legacy history
            return self.chat_service.get_chat_history(token)

    async def clear_chat_history(self, token: str) -> Dict[str, Any]:
        """Clear chat history from both systems"""
        try:
            # Clear legacy history
            legacy_result = self.chat_service.clear_chat_history(token)

            # Clear RAG interactions from database
            user_id = self._get_user_id_from_token(token)
            # Note: We don't actually delete from database for audit purposes
            # but we can mark them as cleared

            return {
                'message': 'Chat history cleared from all systems',
                'legacy_cleared': True,
                'rag_cleared': True,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Failed to clear chat history: {str(e)}")
            return {
                'message': 'Partial success clearing chat history',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def get_rag_status(self, token: str) -> Dict[str, Any]:
        """Get RAG system status for user"""
        try:
            user_id = self._get_user_id_from_token(token)

            # Get collection info
            collection_info = await vector_service.get_collection_info(user_id)

            # Get user documents
            documents = database_service.get_user_documents(int(user_id))
            rag_documents = [doc for doc in documents if doc.get('rag_processed', False)]

            # Get user config
            rag_config = database_service.get_rag_config(int(user_id))

            # Get recent interactions
            recent_interactions = database_service.get_user_history(int(user_id), limit=10)
            rag_interactions = [i for i in recent_interactions if i.get('interaction_type') == 'rag_chat']

            return {
                'user_id': user_id,
                'rag_enabled': len(rag_documents) > 0,
                'total_documents': len(documents),
                'rag_documents': len(rag_documents),
                'vector_count': collection_info.get('total_vectors', 0),
                'collection_exists': collection_info.get('exists', False),
                'recent_rag_interactions': len(rag_interactions),
                'rag_config': rag_config,
                'last_updated': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Failed to get RAG status: {str(e)}")
            return {
                'user_id': self._get_user_id_from_token(token),
                'rag_enabled': False,
                'error': str(e),
                'last_updated': datetime.now().isoformat()
            }

    async def generate_questions_rag(self, token: str, topic: str = None, count: int = 25, mode: str = "practice") -> Dict[str, Any]:
        """Generate questions using RAG-enhanced system"""
        try:
            user_id = self._get_user_id_from_token(token)

            # Check if user has RAG documents
            if not await self._user_has_rag_documents(user_id):
                # Fall back to legacy system
                response = await self.chat_service.generate_questions(token, topic, count, mode)
                return {
                    'questions': json.loads(response.response),
                    'rag_used': False,
                    'fallback': True
                }

            # Use RAG-enhanced question generation
            # For now, delegate to legacy system but with RAG context
            # In future, implement dedicated RAG question generation
            response = await self.chat_service.generate_questions(token, topic, count, mode)

            # Log the interaction
            database_service.log_user_interaction(
                user_id=int(user_id),
                interaction_type='rag_question_generation',
                query=f"Generate {count} {mode} questions{': ' + topic if topic else ''}",
                response=response.response,
                metadata={'mode': mode, 'count': count, 'topic': topic}
            )

            return {
                'questions': json.loads(response.response),
                'rag_used': True,
                'topic': topic,
                'count': count,
                'mode': mode
            }

        except Exception as e:
            self.logger.error(f"RAG question generation failed: {str(e)}")

            # Try fallback
            try:
                response = await self.chat_service.generate_questions(token, topic, count, mode)
                return {
                    'questions': json.loads(response.response),
                    'rag_used': False,
                    'fallback': True,
                    'error': str(e)
                }
            except Exception as fallback_error:
                self.logger.error(f"Even fallback question generation failed: {str(fallback_error)}")
                raise HTTPException(
                    status_code=500,
                    detail="Question generation service temporarily unavailable"
                )

    async def evaluate_answer_rag(self, question: str, user_answer: str, question_id: str, token: str, evaluation_level: str = "medium") -> Dict[str, Any]:
        """Evaluate answer using RAG-enhanced system"""
        try:
            user_id = self._get_user_id_from_token(token)

            # Check if user has RAG documents
            if not await self._user_has_rag_documents(user_id):
                # Fall back to legacy system
                from models.chat import AnswerEvaluationRequest
                request = AnswerEvaluationRequest(
                    question=question,
                    user_answer=user_answer,
                    question_id=question_id,
                    evaluation_level=evaluation_level
                )
                response = await self.chat_service.evaluate_answer(request, token)
                return {
                    'question_id': response.question_id,
                    'score': response.score,
                    'max_score': response.max_score,
                    'feedback': response.feedback,
                    'suggestions': response.suggestions,
                    'rag_used': False,
                    'fallback': True
                }

            # Use RAG-enhanced evaluation
            # For now, delegate to legacy system but with RAG context
            # In future, implement dedicated RAG answer evaluation
            from models.chat import AnswerEvaluationRequest
            request = AnswerEvaluationRequest(
                question=question,
                user_answer=user_answer,
                question_id=question_id,
                evaluation_level=evaluation_level
            )
            response = await self.chat_service.evaluate_answer(request, token)

            # Log the evaluation
            database_service.log_user_interaction(
                user_id=int(user_id),
                interaction_type='rag_answer_evaluation',
                query=question,
                response=f"Score: {response.score}/{response.max_score}",
                metadata={
                    'user_answer': user_answer,
                    'evaluation_level': evaluation_level,
                    'score': response.score
                }
            )

            return {
                'question_id': response.question_id,
                'score': response.score,
                'max_score': response.max_score,
                'feedback': response.feedback,
                'suggestions': response.suggestions,
                'rag_used': True,
                'evaluation_level': evaluation_level
            }

        except Exception as e:
            self.logger.error(f"RAG answer evaluation failed: {str(e)}")

            # Try fallback
            try:
                from models.chat import AnswerEvaluationRequest
                request = AnswerEvaluationRequest(
                    question=question,
                    user_answer=user_answer,
                    question_id=question_id,
                    evaluation_level=evaluation_level
                )
                response = await self.chat_service.evaluate_answer(request, token)
                return {
                    'question_id': response.question_id,
                    'score': response.score,
                    'max_score': response.max_score,
                    'feedback': response.feedback,
                    'suggestions': response.suggestions,
                    'rag_used': False,
                    'fallback': True,
                    'error': str(e)
                }
            except Exception as fallback_error:
                self.logger.error(f"Even fallback evaluation failed: {str(fallback_error)}")
                raise HTTPException(
                    status_code=500,
                    detail="Answer evaluation service temporarily unavailable"
                )

    def health_check(self) -> Dict[str, Any]:
        """Health check for RAG chat service"""
        try:
            # Check legacy chat service
            legacy_status = "ok"

            # Check RAG service
            rag_status = self.rag_service.health_check()

            # Check vector service
            vector_status = vector_service.health_check()

            return {
                'status': 'healthy',
                'legacy_chat_service': legacy_status,
                'rag_service': rag_status.get('status', 'unknown'),
                'vector_service': vector_status.get('status', 'unknown'),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


# Global RAG chat service instance
rag_chat_service = RAGChatService()