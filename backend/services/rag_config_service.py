"""
RAG Configuration Management Service
Production-grade configuration management for RAG system with user preferences and system defaults
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, validator
from fastapi import HTTPException

from backend.config.qdrant_config import qdrant_config
from backend.services.database_service import database_service
from backend.utils.logger import setup_logger


class SystemRAGConfig(BaseModel):
    """System-wide RAG configuration"""
    chunk_size: int = qdrant_config.DEFAULT_CHUNK_SIZE
    chunk_overlap: int = qdrant_config.DEFAULT_CHUNK_OVERLAP
    embedding_model: str = qdrant_config.EMBEDDING_MODEL
    retrieval_top_k: int = qdrant_config.DEFAULT_SEARCH_LIMIT
    search_score_threshold: float = qdrant_config.SEARCH_SCORE_THRESHOLD
    rerank_enabled: bool = True
    max_context_length: int = 4000
    enable_hybrid_search: bool = False
    enable_query_expansion: bool = True
    enable_reranking: bool = True

    @validator('chunk_size')
    def validate_chunk_size(cls, v):
        if v < 100 or v > 5000:
            raise ValueError('Chunk size must be between 100 and 5000')
        return v

    @validator('chunk_overlap')
    def validate_chunk_overlap(cls, v):
        if v < 0 or v > 1000:
            raise ValueError('Chunk overlap must be between 0 and 1000')
        return v

    @validator('retrieval_top_k')
    def validate_retrieval_top_k(cls, v):
        if v < 1 or v > 20:
            raise ValueError('Retrieval top-k must be between 1 and 20')
        return v

    @validator('search_score_threshold')
    def validate_search_score_threshold(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError('Search score threshold must be between 0.0 and 1.0')
        return v


class RAGConfigService:
    """Production-grade RAG configuration management"""

    def __init__(self):
        self.logger = setup_logger(__name__)

        # System default configuration
        self.system_config = SystemRAGConfig()

        # Load configuration from database if available
        self._load_system_config()

    def _load_system_config(self):
        """Load system configuration from database"""
        try:
            # Try to load from a system_config table or use defaults
            # For now, use the pydantic defaults
            pass
        except Exception as e:
            self.logger.warning(f"Failed to load system config from database: {str(e)}")

    def get_system_config(self) -> Dict[str, Any]:
        """Get current system RAG configuration"""
        try:
            return {
                'config': self.system_config.dict(),
                'last_updated': datetime.now().isoformat(),
                'source': 'system_defaults'
            }
        except Exception as e:
            self.logger.error(f"Failed to get system config: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to retrieve system configuration")

    def update_system_config(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """Update system RAG configuration"""
        try:
            # Validate new configuration
            validated_config = SystemRAGConfig(**new_config)

            # Update system config
            self.system_config = validated_config

            # Log the change
            self.logger.info(f"System RAG configuration updated: {new_config}")

            # In production, save to database
            # database_service.log_system_metric('system_config_update', 1, new_config)

            return {
                'message': 'System configuration updated successfully',
                'config': self.system_config.dict(),
                'updated_at': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Failed to update system config: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Configuration update failed: {str(e)}")

    def get_user_config(self, user_id: int) -> Dict[str, Any]:
        """Get RAG configuration for a specific user"""
        try:
            user_config = database_service.get_rag_config(user_id)

            if not user_config:
                # Return system defaults for the user
                return {
                    'user_id': user_id,
                    'config': self.system_config.dict(),
                    'source': 'system_defaults',
                    'is_custom': False,
                    'last_updated': datetime.now().isoformat()
                }

            # Merge user config with system defaults
            merged_config = self._merge_config_with_defaults(user_config)

            return {
                'user_id': user_id,
                'config': merged_config,
                'source': 'user_custom',
                'is_custom': True,
                'user_config': user_config,
                'last_updated': user_config.get('updated_at', datetime.now().isoformat())
            }

        except Exception as e:
            self.logger.error(f"Failed to get user config for user {user_id}: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to retrieve user configuration")

    def update_user_config(self, user_id: int, config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update RAG configuration for a specific user"""
        try:
            # Get current user config
            current_config = database_service.get_rag_config(user_id)

            if not current_config:
                # Create new config with system defaults
                default_config = self.system_config.dict()
                default_config.update(config_updates)
                database_service.update_rag_config(user_id, default_config)
            else:
                # Update existing config
                database_service.update_rag_config(user_id, config_updates)

            # Get updated config
            updated_config = database_service.get_rag_config(user_id)

            # Merge with system defaults
            merged_config = self._merge_config_with_defaults(updated_config)

            self.logger.info(f"User RAG configuration updated for user {user_id}: {config_updates}")

            return {
                'user_id': user_id,
                'message': 'User configuration updated successfully',
                'config': merged_config,
                'updated_config': updated_config,
                'updated_at': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Failed to update user config for user {user_id}: {str(e)}")
            raise HTTPException(status_code=500, detail="Configuration update failed")

    def _merge_config_with_defaults(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user configuration with system defaults"""
        system_config_dict = self.system_config.dict()

        # Start with system defaults
        merged = system_config_dict.copy()

        # Override with user settings
        for key, value in user_config.items():
            if key in merged and value is not None:
                merged[key] = value

        return merged

    def reset_user_config(self, user_id: int) -> Dict[str, Any]:
        """Reset user configuration to system defaults"""
        try:
            # Delete user config (will fall back to system defaults)
            # In a real implementation, you might have a separate table for user configs
            # For now, we'll update with system defaults
            system_config = self.system_config.dict()
            database_service.update_rag_config(user_id, system_config)

            self.logger.info(f"User RAG configuration reset to defaults for user {user_id}")

            return {
                'user_id': user_id,
                'message': 'Configuration reset to system defaults',
                'config': system_config,
                'reset_at': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Failed to reset user config for user {user_id}: {str(e)}")
            raise HTTPException(status_code=500, detail="Configuration reset failed")

    def get_config_recommendations(self, user_id: int) -> Dict[str, Any]:
        """Get configuration recommendations based on user usage patterns"""
        try:
            # Get user analytics
            user_analytics = database_service.get_user_analytics(user_id, days=30)

            # Get user interaction history
            user_history = database_service.get_user_history(user_id, limit=100)

            # Analyze usage patterns
            recommendations = self._analyze_usage_patterns(user_analytics, user_history)

            return {
                'user_id': user_id,
                'recommendations': recommendations,
                'based_on': {
                    'analytics_period_days': 30,
                    'history_items_analyzed': len(user_history),
                    'total_interactions': user_analytics.get('total_interactions', 0)
                },
                'generated_at': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Failed to get recommendations for user {user_id}: {str(e)}")
            return {
                'user_id': user_id,
                'recommendations': [],
                'error': str(e),
                'generated_at': datetime.now().isoformat()
            }

    def _analyze_usage_patterns(self, analytics: Dict[str, Any], history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze user usage patterns and generate recommendations"""
        recommendations = []

        try:
            # Analyze response times
            avg_response_time = analytics.get('average_response_time', 0)
            if avg_response_time > 5.0:  # Slow responses
                recommendations.append({
                    'type': 'performance',
                    'category': 'response_time',
                    'priority': 'high',
                    'current_value': avg_response_time,
                    'suggested_config': {
                        'retrieval_top_k': 3,  # Reduce for faster responses
                        'search_score_threshold': 0.6  # Lower threshold for more results
                    },
                    'reason': 'High response times detected. Reducing retrieval complexity may improve performance.',
                    'expected_benefit': 'Faster response times with slightly reduced accuracy'
                })

            # Analyze interaction types
            interaction_stats = analytics.get('interaction_stats', {})
            chat_interactions = interaction_stats.get('rag_chat', 0)
            qa_interactions = interaction_stats.get('rag_question_generation', 0)

            if chat_interactions > qa_interactions * 2:  # Heavy chat user
                recommendations.append({
                    'type': 'usage_pattern',
                    'category': 'chat_focused',
                    'priority': 'medium',
                    'suggested_config': {
                        'chunk_size': 800,  # Smaller chunks for chat
                        'chunk_overlap': 100,  # Less overlap for chat
                        'retrieval_top_k': 3
                    },
                    'reason': 'User primarily uses chat functionality. Smaller chunks and faster retrieval would be more suitable.',
                    'expected_benefit': 'More responsive chat experience'
                })

            elif qa_interactions > chat_interactions * 2:  # Heavy Q&A user
                recommendations.append({
                    'type': 'usage_pattern',
                    'category': 'qa_focused',
                    'priority': 'medium',
                    'suggested_config': {
                        'chunk_size': 1500,  # Larger chunks for comprehensive Q&A
                        'chunk_overlap': 300,  # More overlap for context
                        'retrieval_top_k': 7  # More results for Q&A
                    },
                    'reason': 'User primarily generates questions and evaluations. Larger context and more comprehensive retrieval would be beneficial.',
                    'expected_benefit': 'Better question generation and more accurate evaluations'
                })

            # Analyze document types (if available in history)
            document_types = {}
            for interaction in history:
                metadata = interaction.get('metadata', {})
                if isinstance(metadata, dict):
                    doc_type = metadata.get('document_type', 'unknown')
                    document_types[doc_type] = document_types.get(doc_type, 0) + 1

            if document_types:
                most_common_type = max(document_types.items(), key=lambda x: x[1])[0]
                if most_common_type in ['research_paper', 'technical_doc']:
                    recommendations.append({
                        'type': 'content_type',
                        'category': 'technical_content',
                        'priority': 'medium',
                        'suggested_config': {
                            'chunk_size': 1200,
                            'search_score_threshold': 0.8,  # Higher threshold for technical content
                            'rerank_enabled': True
                        },
                        'reason': f'Mostly processing {most_common_type.replace("_", " ")}. Technical content benefits from higher precision.',
                        'expected_benefit': 'More accurate results for technical content'
                    })

            # General optimization recommendations
            if not recommendations:
                recommendations.append({
                    'type': 'general',
                    'category': 'optimization',
                    'priority': 'low',
                    'suggested_config': {
                        'enable_query_expansion': True,
                        'enable_reranking': True
                    },
                    'reason': 'Enable advanced features for better overall performance',
                    'expected_benefit': 'Improved retrieval quality and relevance'
                })

        except Exception as e:
            self.logger.warning(f"Error in usage pattern analysis: {str(e)}")

        return recommendations

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate RAG configuration parameters"""
        try:
            validated_config = SystemRAGConfig(**config)

            # Additional validation
            validation_results = {
                'valid': True,
                'warnings': [],
                'errors': []
            }

            # Performance warnings
            if config.get('chunk_size', 0) > 2000:
                validation_results['warnings'].append({
                    'parameter': 'chunk_size',
                    'message': 'Large chunk sizes may impact performance and accuracy',
                    'suggestion': 'Consider using chunk sizes between 500-1500 for optimal performance'
                })

            if config.get('retrieval_top_k', 0) > 10:
                validation_results['warnings'].append({
                    'parameter': 'retrieval_top_k',
                    'message': 'High retrieval top-k values may slow down responses',
                    'suggestion': 'Consider using values between 3-7 for most use cases'
                })

            if config.get('search_score_threshold', 1.0) > 0.9:
                validation_results['warnings'].append({
                    'parameter': 'search_score_threshold',
                    'message': 'Very high score threshold may exclude relevant results',
                    'suggestion': 'Consider using thresholds between 0.6-0.8 for better recall'
                })

            # Chunk overlap validation
            chunk_size = config.get('chunk_size', qdrant_config.DEFAULT_CHUNK_SIZE)
            chunk_overlap = config.get('chunk_overlap', qdrant_config.DEFAULT_CHUNK_OVERLAP)

            if chunk_overlap > chunk_size * 0.5:
                validation_results['warnings'].append({
                    'parameter': 'chunk_overlap',
                    'message': 'Chunk overlap is more than 50% of chunk size',
                    'suggestion': 'Consider reducing overlap to 10-30% of chunk size'
                })

            return {
                'config': validated_config.dict(),
                'validation': validation_results
            }

        except Exception as e:
            return {
                'config': config,
                'validation': {
                    'valid': False,
                    'errors': [str(e)],
                    'warnings': []
                }
            }

    def get_config_presets(self) -> Dict[str, Any]:
        """Get predefined configuration presets for different use cases"""
        presets = {
            'fast_chat': {
                'name': 'Fast Chat',
                'description': 'Optimized for quick chat responses',
                'config': {
                    'chunk_size': 800,
                    'chunk_overlap': 100,
                    'retrieval_top_k': 3,
                    'search_score_threshold': 0.7,
                    'enable_query_expansion': False,
                    'enable_reranking': False
                },
                'use_case': 'Quick conversational responses'
            },
            'comprehensive_qa': {
                'name': 'Comprehensive Q&A',
                'description': 'Optimized for detailed question generation and evaluation',
                'config': {
                    'chunk_size': 1500,
                    'chunk_overlap': 300,
                    'retrieval_top_k': 7,
                    'search_score_threshold': 0.6,
                    'enable_query_expansion': True,
                    'enable_reranking': True
                },
                'use_case': 'Detailed question generation and answer evaluation'
            },
            'technical_documents': {
                'name': 'Technical Documents',
                'description': 'Optimized for research papers and technical content',
                'config': {
                    'chunk_size': 1200,
                    'chunk_overlap': 200,
                    'retrieval_top_k': 5,
                    'search_score_threshold': 0.8,
                    'enable_query_expansion': True,
                    'enable_reranking': True
                },
                'use_case': 'Technical papers, research documents, academic content'
            },
            'balanced': {
                'name': 'Balanced',
                'description': 'Balanced configuration for mixed use cases',
                'config': {
                    'chunk_size': 1000,
                    'chunk_overlap': 200,
                    'retrieval_top_k': 5,
                    'search_score_threshold': 0.7,
                    'enable_query_expansion': True,
                    'enable_reranking': True
                },
                'use_case': 'General purpose, mixed usage patterns'
            }
        }

        return {
            'presets': presets,
            'recommended_for_new_users': 'balanced',
            'last_updated': datetime.now().isoformat()
        }

    def health_check(self) -> Dict[str, Any]:
        """Health check for configuration service"""
        try:
            return {
                'status': 'healthy',
                'system_config_loaded': True,
                'config_validation': 'ok',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


# Global RAG configuration service instance
rag_config_service = RAGConfigService()