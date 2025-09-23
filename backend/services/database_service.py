"""
SQLite Database Service for RAG System
Production-grade database operations with comprehensive error handling and migrations
"""

import sqlite3
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import threading
from contextlib import contextmanager

from utils.logger import get_logger


class DatabaseError(Exception):
    """Custom exception for database errors"""
    pass


class DatabaseService:
    """Production-grade SQLite database service with connection pooling"""

    def __init__(self, db_path: str = "backend/database/lumina_iq_rag.db"):
        self.db_path = Path(db_path)
        self.logger = get_logger(__name__)

        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Connection pool for thread safety
        self._local = threading.local()
        self._initialize_database()

    @property
    def connection(self):
        """Get thread-local database connection"""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                isolation_level=None  # Enable autocommit mode
            )
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection

    @contextmanager
    def get_cursor(self):
        """Context manager for database cursor"""
        cursor = self.connection.cursor()
        try:
            yield cursor
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            raise e
        finally:
            cursor.close()

    def _initialize_database(self):
        """Initialize database with all required tables"""
        try:
            with self.get_cursor() as cursor:
                # Users table (enhanced with RAG features)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        email TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        rag_enabled BOOLEAN DEFAULT FALSE,
                        embedding_quota INTEGER DEFAULT 1000000,
                        quota_used INTEGER DEFAULT 0,
                        subscription_tier TEXT DEFAULT 'free',
                        last_login TIMESTAMP,
                        is_active BOOLEAN DEFAULT TRUE
                    )
                ''')

                # Documents table (enhanced with RAG processing)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS documents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                        filename TEXT NOT NULL,
                        original_filename TEXT NOT NULL,
                        file_path TEXT,
                        file_size INTEGER,
                        total_pages INTEGER,
                        total_tokens INTEGER,
                        rag_processed BOOLEAN DEFAULT FALSE,
                        processing_status TEXT DEFAULT 'pending',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata JSON
                    )
                ''')

                # RAG Chunks table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS rag_chunks (
                        id TEXT PRIMARY KEY,
                        document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                        content TEXT NOT NULL,
                        chunk_type TEXT DEFAULT 'paragraph',
                        importance_score REAL DEFAULT 0.5,
                        token_count INTEGER,
                        start_position INTEGER,
                        end_position INTEGER,
                        metadata JSON,
                        embedding_status TEXT DEFAULT 'pending',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # User Interactions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_interactions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                        interaction_type TEXT NOT NULL,
                        query TEXT,
                        response TEXT,
                        rag_chunks_used JSON,
                        performance_score REAL,
                        response_time REAL,
                        tokens_used INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata JSON
                    )
                ''')

                # RAG Configuration table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS rag_config (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                        chunk_size INTEGER DEFAULT 1000,
                        chunk_overlap INTEGER DEFAULT 200,
                        embedding_model TEXT DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
                        retrieval_top_k INTEGER DEFAULT 5,
                        rerank_enabled BOOLEAN DEFAULT TRUE,
                        search_score_threshold REAL DEFAULT 0.7,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # User Sessions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                        session_token TEXT UNIQUE NOT NULL,
                        ip_address TEXT,
                        user_agent TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP,
                        is_active BOOLEAN DEFAULT TRUE
                    )
                ''')

                # System Metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        metric_name TEXT NOT NULL,
                        metric_value REAL,
                        metadata JSON,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Create indexes for performance
                self._create_indexes(cursor)

                self.logger.info("Database initialized successfully")

        except Exception as e:
            self.logger.error(f"Database initialization failed: {str(e)}")
            raise DatabaseError(f"Database setup error: {str(e)}")

    def _create_indexes(self, cursor):
        """Create database indexes for optimal performance"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)",
            "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)",
            "CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_documents_rag_processed ON documents(rag_processed)",
            "CREATE INDEX IF NOT EXISTS idx_rag_chunks_document_id ON rag_chunks(document_id)",
            "CREATE INDEX IF NOT EXISTS idx_rag_chunks_user_id ON rag_chunks(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_rag_chunks_embedding_status ON rag_chunks(embedding_status)",
            "CREATE INDEX IF NOT EXISTS idx_user_interactions_user_id ON user_interactions(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_user_interactions_type ON user_interactions(interaction_type)",
            "CREATE INDEX IF NOT EXISTS idx_user_interactions_created_at ON user_interactions(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_rag_config_user_id ON rag_config(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_user_sessions_token ON user_sessions(session_token)",
            "CREATE INDEX IF NOT EXISTS idx_user_sessions_expires_at ON user_sessions(expires_at)",
            "CREATE INDEX IF NOT EXISTS idx_system_metrics_name ON system_metrics(metric_name)",
            "CREATE INDEX IF NOT EXISTS idx_system_metrics_created_at ON system_metrics(created_at)"
        ]

        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
            except Exception as e:
                self.logger.warning(f"Failed to create index: {str(e)}")

    # User Management Methods
    def create_user(self, username: str, email: str, password_hash: str) -> int:
        """Create a new user"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute('''
                    INSERT INTO users (username, email, password_hash, rag_enabled, embedding_quota)
                    VALUES (?, ?, ?, ?, ?)
                ''', (username, email, password_hash, True, 1000000))

                user_id = cursor.lastrowid

                # Create default RAG configuration
                cursor.execute('''
                    INSERT INTO rag_config (user_id)
                    VALUES (?)
                ''', (user_id,))

                self.logger.info(f"User created successfully: {username}")
                return user_id

        except sqlite3.IntegrityError as e:
            if "username" in str(e):
                raise DatabaseError("Username already exists")
            elif "email" in str(e):
                raise DatabaseError("Email already exists")
            else:
                raise DatabaseError(f"User creation failed: {str(e)}")

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute('''
                    SELECT u.*, rc.*
                    FROM users u
                    LEFT JOIN rag_config rc ON u.id = rc.user_id
                    WHERE u.username = ? AND u.is_active = TRUE
                ''', (username,))

                row = cursor.fetchone()
                return dict(row) if row else None

        except Exception as e:
            self.logger.error(f"Failed to get user {username}: {str(e)}")
            return None

    def update_user_quota(self, user_id: int, tokens_used: int) -> bool:
        """Update user token quota usage"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute('''
                    UPDATE users
                    SET quota_used = quota_used + ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (tokens_used, user_id))

                return cursor.rowcount > 0

        except Exception as e:
            self.logger.error(f"Failed to update user quota: {str(e)}")
            return False

    # Document Management Methods
    def create_document(self, user_id: int, filename: str, file_size: int,
                       total_pages: int, total_tokens: int, metadata: Dict = None) -> int:
        """Create a new document record"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute('''
                    INSERT INTO documents (user_id, filename, original_filename, file_size,
                                        total_pages, total_tokens, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (user_id, filename, filename, file_size, total_pages, total_tokens,
                      json.dumps(metadata) if metadata else None))

                document_id = cursor.lastrowid
                self.logger.info(f"Document created: {filename} for user {user_id}")
                return document_id

        except Exception as e:
            self.logger.error(f"Failed to create document: {str(e)}")
            raise DatabaseError(f"Document creation failed: {str(e)}")

    def get_user_documents(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all documents for a user"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute('''
                    SELECT * FROM documents
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                ''', (user_id,))

                return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            self.logger.error(f"Failed to get documents for user {user_id}: {str(e)}")
            return []

    # RAG Chunks Management Methods
    def create_chunk(self, chunk_id: str, document_id: int, user_id: int, content: str,
                    chunk_type: str, importance_score: float, token_count: int,
                    start_position: int, end_position: int, metadata: Dict = None) -> bool:
        """Create a new RAG chunk"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute('''
                    INSERT INTO rag_chunks (id, document_id, user_id, content, chunk_type,
                                         importance_score, token_count, start_position,
                                         end_position, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (chunk_id, document_id, user_id, content, chunk_type, importance_score,
                      token_count, start_position, end_position,
                      json.dumps(metadata) if metadata else None))

                return True

        except Exception as e:
            self.logger.error(f"Failed to create chunk {chunk_id}: {str(e)}")
            return False

    def get_document_chunks(self, document_id: int) -> List[Dict[str, Any]]:
        """Get all chunks for a document"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute('''
                    SELECT * FROM rag_chunks
                    WHERE document_id = ?
                    ORDER BY start_position ASC
                ''', (document_id,))

                return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            self.logger.error(f"Failed to get chunks for document {document_id}: {str(e)}")
            return []

    def update_chunk_embedding_status(self, chunk_id: str, status: str) -> bool:
        """Update embedding status of a chunk"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute('''
                    UPDATE rag_chunks
                    SET embedding_status = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (status, chunk_id))

                return cursor.rowcount > 0

        except Exception as e:
            self.logger.error(f"Failed to update chunk status {chunk_id}: {str(e)}")
            return False

    # User Interactions Methods
    def log_user_interaction(self, user_id: int, interaction_type: str, query: str = None,
                           response: str = None, rag_chunks_used: List = None,
                           performance_score: float = None, response_time: float = None,
                           tokens_used: int = None, metadata: Dict = None) -> int:
        """Log user interaction"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute('''
                    INSERT INTO user_interactions (user_id, interaction_type, query, response,
                                                 rag_chunks_used, performance_score, response_time,
                                                 tokens_used, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (user_id, interaction_type, query, response,
                      json.dumps(rag_chunks_used) if rag_chunks_used else None,
                      performance_score, response_time, tokens_used,
                      json.dumps(metadata) if metadata else None))

                interaction_id = cursor.lastrowid

                # Update user quota if tokens were used
                if tokens_used:
                    self.update_user_quota(user_id, tokens_used)

                self.logger.info(f"Logged interaction {interaction_id} for user {user_id}")
                return interaction_id

        except Exception as e:
            self.logger.error(f"Failed to log interaction for user {user_id}: {str(e)}")
            return 0

    def get_user_history(self, user_id: int, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Get user interaction history"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute('''
                    SELECT * FROM user_interactions
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                ''', (user_id, limit, offset))

                return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            self.logger.error(f"Failed to get history for user {user_id}: {str(e)}")
            return []

    # RAG Configuration Methods
    def get_rag_config(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get RAG configuration for a user"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute('''
                    SELECT * FROM rag_config
                    WHERE user_id = ?
                ''', (user_id,))

                row = cursor.fetchone()
                return dict(row) if row else None

        except Exception as e:
            self.logger.error(f"Failed to get RAG config for user {user_id}: {str(e)}")
            return None

    def update_rag_config(self, user_id: int, config_updates: Dict[str, Any]) -> bool:
        """Update RAG configuration for a user"""
        try:
            with self.get_cursor() as cursor:
                # Build dynamic update query
                set_clauses = []
                values = []

                for key, value in config_updates.items():
                    if hasattr(value, '__dict__'):  # Handle Pydantic models
                        continue
                    set_clauses.append(f"{key} = ?")
                    values.append(value)

                values.append(user_id)

                query = f'''
                    UPDATE rag_config
                    SET {', '.join(set_clauses)}, updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                '''

                cursor.execute(query, values)
                return cursor.rowcount > 0

        except Exception as e:
            self.logger.error(f"Failed to update RAG config for user {user_id}: {str(e)}")
            return False

    # Analytics and Metrics Methods
    def log_system_metric(self, metric_name: str, metric_value: float, metadata: Dict = None) -> bool:
        """Log system metric"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute('''
                    INSERT INTO system_metrics (metric_name, metric_value, metadata)
                    VALUES (?, ?, ?)
                ''', (metric_name, metric_value, json.dumps(metadata) if metadata else None))

                return True

        except Exception as e:
            self.logger.error(f"Failed to log system metric {metric_name}: {str(e)}")
            return False

    def get_user_analytics(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """Get user analytics for the specified period"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)

            with self.get_cursor() as cursor:
                # Get interaction counts by type
                cursor.execute('''
                    SELECT interaction_type, COUNT(*) as count
                    FROM user_interactions
                    WHERE user_id = ? AND created_at >= ?
                    GROUP BY interaction_type
                ''', (user_id, cutoff_date.isoformat()))

                interaction_stats = {row['interaction_type']: row['count'] for row in cursor.fetchall()}

                # Get total tokens used
                cursor.execute('''
                    SELECT SUM(tokens_used) as total_tokens
                    FROM user_interactions
                    WHERE user_id = ? AND created_at >= ? AND tokens_used IS NOT NULL
                ''', (user_id, cutoff_date.isoformat()))

                tokens_row = cursor.fetchone()
                total_tokens = tokens_row['total_tokens'] or 0

                # Get average response time
                cursor.execute('''
                    SELECT AVG(response_time) as avg_response_time
                    FROM user_interactions
                    WHERE user_id = ? AND created_at >= ? AND response_time IS NOT NULL
                ''', (user_id, cutoff_date.isoformat()))

                response_time_row = cursor.fetchone()
                avg_response_time = response_time_row['avg_response_time'] or 0

                return {
                    'user_id': user_id,
                    'period_days': days,
                    'interaction_stats': interaction_stats,
                    'total_tokens_used': total_tokens,
                    'average_response_time': avg_response_time,
                    'generated_at': datetime.now().isoformat()
                }

        except Exception as e:
            self.logger.error(f"Failed to get analytics for user {user_id}: {str(e)}")
            return {}

    def cleanup_old_data(self, days_to_keep: int = 90) -> Dict[str, int]:
        """Clean up old data for maintenance"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)

            with self.get_cursor() as cursor:
                # Clean up old user interactions
                cursor.execute('''
                    DELETE FROM user_interactions
                    WHERE created_at < ?
                ''', (cutoff_date.isoformat(),))

                interactions_deleted = cursor.rowcount

                # Clean up old system metrics
                cursor.execute('''
                    DELETE FROM system_metrics
                    WHERE created_at < ?
                ''', (cutoff_date.isoformat(),))

                metrics_deleted = cursor.rowcount

                self.logger.info(f"Cleaned up {interactions_deleted} old interactions and {metrics_deleted} old metrics")

                return {
                    'interactions_deleted': interactions_deleted,
                    'metrics_deleted': metrics_deleted,
                    'cutoff_date': cutoff_date.isoformat()
                }

        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {str(e)}")
            return {}

    def health_check(self) -> Dict[str, Any]:
        """Database health check"""
        try:
            with self.get_cursor() as cursor:
                # Test basic operations
                cursor.execute("SELECT COUNT(*) FROM users")
                user_count = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM documents")
                document_count = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM rag_chunks")
                chunk_count = cursor.fetchone()[0]

                # Check database file size
                db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

                return {
                    'status': 'healthy',
                    'database_path': str(self.db_path),
                    'database_size_mb': db_size / (1024 * 1024),
                    'user_count': user_count,
                    'document_count': document_count,
                    'chunk_count': chunk_count,
                    'timestamp': datetime.now().isoformat()
                }

        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


# Global database service instance
database_service = DatabaseService()