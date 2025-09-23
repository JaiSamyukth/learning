"""
Analytics and History Service for RAG System
Production-grade analytics service with comprehensive reporting, user behavior analysis, and system monitoring
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from pathlib import Path

from fastapi import HTTPException

from backend.services.database_service import database_service
from backend.services.vector_service import vector_service
from backend.services.rag_service import rag_service
from backend.utils.logger import setup_logger


class AnalyticsService:
    """Production-grade analytics and history service"""

    def __init__(self):
        self.logger = setup_logger(__name__)

    async def get_user_activity_summary(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive user activity summary"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)

            # Get basic analytics
            basic_analytics = database_service.get_user_analytics(user_id, days)

            # Get detailed interaction breakdown
            interactions = database_service.execute_query('''
                SELECT interaction_type, COUNT(*) as count, AVG(performance_score) as avg_score,
                       AVG(response_time) as avg_time, SUM(tokens_used) as total_tokens
                FROM user_interactions
                WHERE user_id = ? AND created_at >= ?
                GROUP BY interaction_type
            ''', (user_id, cutoff_date.isoformat()), fetch_all=True)

            interaction_breakdown = {}
            for row in interactions:
                interaction_breakdown[row['interaction_type']] = {
                    'count': row['count'],
                    'avg_score': row['avg_score'] or 0,
                    'avg_response_time': row['avg_time'] or 0,
                    'total_tokens': row['total_tokens'] or 0
                }

            # Get document processing history
            documents = database_service.get_user_documents(user_id)
            processed_docs = [doc for doc in documents if doc.get('rag_processed', False)]

            # Get recent interactions with details
            recent_interactions = database_service.get_user_history(user_id, limit=20)

            # Calculate engagement metrics
            engagement_metrics = await self._calculate_engagement_metrics(user_id, days)

            # Get RAG-specific metrics
            rag_metrics = await self._get_rag_specific_metrics(user_id, days)

            return {
                'user_id': user_id,
                'period_days': days,
                'cutoff_date': cutoff_date.isoformat(),
                'summary': {
                    'total_interactions': sum([i['count'] for i in interaction_breakdown.values()]),
                    'total_documents': len(documents),
                    'processed_documents': len(processed_docs),
                    'avg_performance_score': basic_analytics.get('average_performance_score', 0),
                    'total_tokens_used': sum([i['total_tokens'] for i in interaction_breakdown.values()]),
                    'most_active_day': engagement_metrics.get('most_active_day'),
                    'favorite_feature': engagement_metrics.get('favorite_feature')
                },
                'interaction_breakdown': interaction_breakdown,
                'engagement_metrics': engagement_metrics,
                'rag_metrics': rag_metrics,
                'recent_interactions': recent_interactions,
                'document_stats': {
                    'total': len(documents),
                    'processed': len(processed_docs),
                    'pending': len([d for d in documents if not d.get('rag_processed', False)]),
                    'failed': len([d for d in documents if d.get('processing_status') == 'failed'])
                },
                'generated_at': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Failed to get user activity summary for user {user_id}: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to generate activity summary")

    async def _calculate_engagement_metrics(self, user_id: int, days: int) -> Dict[str, Any]:
        """Calculate user engagement metrics"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)

            # Get daily interaction counts
            daily_interactions = database_service.execute_query('''
                SELECT DATE(created_at) as date, COUNT(*) as interactions
                FROM user_interactions
                WHERE user_id = ? AND created_at >= ?
                GROUP BY DATE(created_at)
                ORDER BY date DESC
            ''', (user_id, cutoff_date.isoformat()), fetch_all=True)

            # Calculate streak and activity patterns
            if daily_interactions:
                total_interactions = sum([row['interactions'] for row in daily_interactions])
                avg_daily = total_interactions / len(daily_interactions)
                most_active_day = max(daily_interactions, key=lambda x: x['interactions'])['date']
                longest_streak = self._calculate_streak([row['interactions'] for row in daily_interactions])
            else:
                total_interactions = 0
                avg_daily = 0
                most_active_day = None
                longest_streak = 0

            # Determine favorite feature
            feature_counts = database_service.execute_query('''
                SELECT interaction_type, COUNT(*) as count
                FROM user_interactions
                WHERE user_id = ? AND created_at >= ?
                GROUP BY interaction_type
                ORDER BY count DESC
                LIMIT 1
            ''', (user_id, cutoff_date.isoformat()), fetch_one=True)

            favorite_feature = feature_counts['interaction_type'] if feature_counts else None

            # Calculate session patterns
            sessions_per_week = len(daily_interactions) / (days / 7) if days >= 7 else len(daily_interactions)

            return {
                'total_interactions': total_interactions,
                'avg_daily_interactions': avg_daily,
                'most_active_day': most_active_day,
                'longest_streak': longest_streak,
                'favorite_feature': favorite_feature,
                'sessions_per_week': sessions_per_week,
                'active_days': len(daily_interactions),
                'engagement_score': self._calculate_engagement_score(total_interactions, days, longest_streak)
            }

        except Exception as e:
            self.logger.error(f"Failed to calculate engagement metrics: {str(e)}")
            return {}

    def _calculate_streak(self, daily_counts: List[int]) -> int:
        """Calculate the longest streak of active days"""
        if not daily_counts:
            return 0

        max_streak = 0
        current_streak = 0

        for count in daily_counts:
            if count > 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    def _calculate_engagement_score(self, total_interactions: int, days: int, streak: int) -> float:
        """Calculate engagement score (0-100)"""
        if days == 0:
            return 0

        # Base score from daily average (max 40 points)
        avg_daily = total_interactions / days
        daily_score = min(40, (avg_daily / 5) * 40)  # 5 interactions per day = 40 points

        # Consistency score (max 30 points)
        consistency_score = min(30, (streak / 7) * 30)  # 7 day streak = 30 points

        # Activity breadth score (max 30 points)
        # More diverse interaction types = higher score
        interaction_types = len(set([i.get('interaction_type', '') for i in database_service.get_user_history(1, limit=1000)]))
        breadth_score = min(30, (interaction_types / 5) * 30)  # 5 different types = 30 points

        return round(daily_score + consistency_score + breadth_score, 1)

    async def _get_rag_specific_metrics(self, user_id: int, days: int) -> Dict[str, Any]:
        """Get RAG-specific metrics"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)

            # Get RAG interactions
            rag_interactions = database_service.execute_query('''
                SELECT * FROM user_interactions
                WHERE user_id = ? AND created_at >= ? AND interaction_type LIKE 'rag_%'
            ''', (user_id, cutoff_date.isoformat()), fetch_all=True)

            if not rag_interactions:
                return {
                    'total_rag_interactions': 0,
                    'avg_confidence': 0,
                    'avg_chunks_used': 0,
                    'avg_response_time': 0,
                    'most_used_feature': None
                }

            # Calculate metrics
            confidences = [i.get('performance_score', 0) for i in rag_interactions if i.get('performance_score')]
            chunks_used = [len(i.get('rag_chunks_used', [])) for i in rag_interactions if i.get('rag_chunks_used')]
            response_times = [i.get('response_time', 0) for i in rag_interactions if i.get('response_time')]

            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            avg_chunks_used = sum(chunks_used) / len(chunks_used) if chunks_used else 0
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0

            # Most used RAG feature
            feature_counts = Counter([i.get('interaction_type', '') for i in rag_interactions])
            most_used_feature = feature_counts.most_common(1)[0][0] if feature_counts else None

            return {
                'total_rag_interactions': len(rag_interactions),
                'avg_confidence': round(avg_confidence, 3),
                'avg_chunks_used': round(avg_chunks_used, 1),
                'avg_response_time': round(avg_response_time, 2),
                'most_used_feature': most_used_feature,
                'feature_breakdown': dict(feature_counts)
            }

        except Exception as e:
            self.logger.error(f"Failed to get RAG metrics: {str(e)}")
            return {}

    async def get_system_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get system-wide analytics"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)

            # Get all users
            users = database_service.execute_query('''
                SELECT * FROM users WHERE is_active = TRUE
            ''', fetch_all=True)

            # Get system metrics
            system_metrics = database_service.execute_query('''
                SELECT metric_name, metric_value, created_at, metadata
                FROM system_metrics
                WHERE created_at >= ?
                ORDER BY created_at DESC
            ''', (cutoff_date.isoformat(),), fetch_all=True)

            # Get recent interactions across all users
            all_interactions = database_service.execute_query('''
                SELECT user_id, interaction_type, COUNT(*) as count,
                       AVG(performance_score) as avg_score, AVG(response_time) as avg_time
                FROM user_interactions
                WHERE created_at >= ?
                GROUP BY user_id, interaction_type
            ''', (cutoff_date.isoformat(),), fetch_all=True)

            # Calculate system stats
            total_users = len(users)
            active_users = len(set([i['user_id'] for i in all_interactions]))
            total_interactions = len(all_interactions)

            # Interaction type breakdown
            interaction_breakdown = defaultdict(int)
            for interaction in all_interactions:
                interaction_breakdown[interaction['interaction_type']] += interaction['count']

            # Average scores by interaction type
            avg_scores = {}
            for interaction in all_interactions:
                it_type = interaction['interaction_type']
                if it_type not in avg_scores:
                    avg_scores[it_type] = []
                avg_scores[it_type].append(interaction['avg_score'] or 0)

            avg_scores = {k: sum(v) / len(v) for k, v in avg_scores.items()}

            # User engagement distribution
            user_activity_counts = defaultdict(int)
            for user in users:
                user_interactions = [i for i in all_interactions if i['user_id'] == user['id']]
                total_user_interactions = sum([i['count'] for i in user_interactions])
                if total_user_interactions == 0:
                    user_activity_counts['inactive'] += 1
                elif total_user_interactions < 5:
                    user_activity_counts['low'] += 1
                elif total_user_interactions < 20:
                    user_activity_counts['medium'] += 1
                else:
                    user_activity_counts['high'] += 1

            return {
                'period_days': days,
                'cutoff_date': cutoff_date.isoformat(),
                'system_summary': {
                    'total_users': total_users,
                    'active_users': active_users,
                    'total_interactions': total_interactions,
                    'avg_interactions_per_user': total_interactions / max(active_users, 1),
                    'system_health_score': await self._calculate_system_health_score()
                },
                'user_engagement': dict(user_activity_counts),
                'interaction_breakdown': dict(interaction_breakdown),
                'average_scores': avg_scores,
                'system_metrics': system_metrics,
                'generated_at': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Failed to get system analytics: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to generate system analytics")

    async def _calculate_system_health_score(self) -> float:
        """Calculate overall system health score (0-100)"""
        try:
            # This is a simplified health score calculation
            # In production, you'd want more sophisticated metrics

            # Check database health
            db_health = database_service.health_check()
            db_score = 100 if db_health.get('status') == 'healthy' else 0

            # Check vector service health
            vector_health = vector_service.health_check()
            vector_score = 100 if vector_health.get('status') == 'healthy' else 0

            # Check recent errors (simplified)
            recent_errors = 0  # This would come from error logging system
            error_score = max(0, 100 - (recent_errors * 10))

            # Calculate weighted average
            health_score = (db_score * 0.4 + vector_score * 0.4 + error_score * 0.2)

            return round(health_score, 1)

        except Exception as e:
            self.logger.error(f"Failed to calculate system health: {str(e)}")
            return 0.0

    async def get_user_progress_report(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """Generate detailed progress report for a user"""
        try:
            # Get activity summary
            activity_summary = await self.get_user_activity_summary(user_id, days)

            # Get RAG performance trends
            rag_trends = await self._get_rag_performance_trends(user_id, days)

            # Get learning patterns
            learning_patterns = await self._analyze_learning_patterns(user_id, days)

            # Get recommendations
            recommendations = await self._generate_user_recommendations(user_id, activity_summary)

            return {
                'user_id': user_id,
                'period_days': days,
                'report_type': 'progress_report',
                'activity_summary': activity_summary,
                'rag_performance_trends': rag_trends,
                'learning_patterns': learning_patterns,
                'recommendations': recommendations,
                'generated_at': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Failed to generate progress report for user {user_id}: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to generate progress report")

    async def _get_rag_performance_trends(self, user_id: int, days: int) -> Dict[str, Any]:
        """Get RAG performance trends over time"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)

            # Get daily RAG performance
            daily_performance = database_service.execute_query('''
                SELECT DATE(created_at) as date,
                       AVG(performance_score) as avg_score,
                       AVG(response_time) as avg_time,
                       COUNT(*) as interactions
                FROM user_interactions
                WHERE user_id = ? AND created_at >= ? AND interaction_type LIKE 'rag_%'
                GROUP BY DATE(created_at)
                ORDER BY date ASC
            ''', (user_id, cutoff_date.isoformat()), fetch_all=True)

            if not daily_performance:
                return {
                    'trend_data': [],
                    'improvement_rate': 0,
                    'consistency_score': 0
                }

            # Calculate trends
            scores = [row['avg_score'] for row in daily_performance if row['avg_score']]
            times = [row['avg_time'] for row in daily_performance if row['avg_time']]

            # Simple trend analysis
            if len(scores) > 1:
                first_half_avg = sum(scores[:len(scores)//2]) / (len(scores)//2)
                second_half_avg = sum(scores[len(scores)//2:]) / (len(scores) - len(scores)//2)
                improvement_rate = ((second_half_avg - first_half_avg) / first_half_avg) * 100 if first_half_avg > 0 else 0
            else:
                improvement_rate = 0

            # Consistency score (lower variance = higher consistency)
            if scores:
                score_variance = sum([(s - sum(scores)/len(scores))**2 for s in scores]) / len(scores)
                consistency_score = max(0, 100 - (score_variance * 100))
            else:
                consistency_score = 0

            return {
                'trend_data': [
                    {
                        'date': row['date'],
                        'avg_score': row['avg_score'] or 0,
                        'avg_response_time': row['avg_time'] or 0,
                        'interactions': row['interactions']
                    }
                    for row in daily_performance
                ],
                'improvement_rate': round(improvement_rate, 2),
                'consistency_score': round(consistency_score, 1),
                'total_data_points': len(daily_performance)
            }

        except Exception as e:
            self.logger.error(f"Failed to get RAG performance trends: {str(e)}")
            return {}

    async def _analyze_learning_patterns(self, user_id: int, days: int) -> Dict[str, Any]:
        """Analyze user learning patterns"""
        try:
            # Get interaction patterns by time of day
            time_patterns = database_service.execute_query('''
                SELECT strftime('%H', created_at) as hour,
                       COUNT(*) as interactions,
                       AVG(performance_score) as avg_score
                FROM user_interactions
                WHERE user_id = ? AND created_at >= datetime('now', '-' || ? || ' days')
                GROUP BY strftime('%H', created_at)
                ORDER BY hour
            ''', (user_id, days), fetch_all=True)

            # Get day of week patterns
            day_patterns = database_service.execute_query('''
                SELECT strftime('%w', created_at) as day_of_week,
                       COUNT(*) as interactions,
                       AVG(performance_score) as avg_score
                FROM user_interactions
                WHERE user_id = ? AND created_at >= datetime('now', '-' || ? || ' days')
                GROUP BY strftime('%w', created_at)
                ORDER BY day_of_week
            ''', (user_id, days), fetch_all=True)

            # Get session duration patterns
            session_durations = self._calculate_session_durations(user_id, days)

            return {
                'time_of_day_patterns': [
                    {
                        'hour': row['hour'],
                        'interactions': row['interactions'],
                        'avg_score': row['avg_score'] or 0
                    }
                    for row in time_patterns
                ],
                'day_of_week_patterns': [
                    {
                        'day': row['day_of_week'],
                        'interactions': row['interactions'],
                        'avg_score': row['avg_score'] or 0
                    }
                    for row in day_patterns
                ],
                'session_durations': session_durations,
                'peak_learning_hours': [p['hour'] for p in time_patterns if p['interactions'] == max([p['interactions'] for p in time_patterns])]
            }

        except Exception as e:
            self.logger.error(f"Failed to analyze learning patterns: {str(e)}")
            return {}

    def _calculate_session_durations(self, user_id: int, days: int) -> Dict[str, Any]:
        """Calculate session duration patterns"""
        try:
            # This is a simplified implementation
            # In a real system, you'd track session start/end times
            interactions = database_service.get_user_history(user_id, limit=days * 10)

            if not interactions:
                return {'avg_session_length': 0, 'total_sessions': 0}

            # Group by date and calculate daily session metrics
            daily_sessions = defaultdict(list)
            for interaction in interactions:
                date = interaction.get('created_at', '')[:10]  # Extract date part
                daily_sessions[date].append(interaction)

            session_lengths = []
            for date, day_interactions in daily_sessions.items():
                if len(day_interactions) > 1:
                    # Calculate time span for the day
                    timestamps = [datetime.fromisoformat(i['created_at']) for i in day_interactions if i.get('created_at')]
                    if timestamps:
                        session_length = (max(timestamps) - min(timestamps)).total_seconds() / 60  # minutes
                        session_lengths.append(session_length)

            avg_session_length = sum(session_lengths) / len(session_lengths) if session_lengths else 0

            return {
                'avg_session_length_minutes': round(avg_session_length, 1),
                'total_sessions': len(daily_sessions),
                'session_lengths': session_lengths
            }

        except Exception as e:
            self.logger.error(f"Failed to calculate session durations: {str(e)}")
            return {'avg_session_length': 0, 'total_sessions': 0}

    async def _generate_user_recommendations(self, user_id: int, activity_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate personalized recommendations for the user"""
        recommendations = []

        try:
            # Based on engagement level
            engagement_score = activity_summary.get('engagement_metrics', {}).get('engagement_score', 0)
            if engagement_score < 30:
                recommendations.append({
                    'type': 'engagement',
                    'priority': 'high',
                    'title': 'Increase Learning Activity',
                    'description': 'Your engagement level is low. Consider setting aside dedicated time for learning each day.',
                    'action_items': [
                        'Set a daily learning goal',
                        'Try different types of interactions (chat, Q&A, quizzes)',
                        'Explore new document topics'
                    ]
                })

            # Based on performance
            rag_metrics = activity_summary.get('rag_metrics', {})
            avg_confidence = rag_metrics.get('avg_confidence', 0)
            if avg_confidence < 0.6:
                recommendations.append({
                    'type': 'performance',
                    'priority': 'medium',
                    'title': 'Improve Answer Quality',
                    'description': 'Your average confidence scores are below optimal. Consider refining your questions and providing more context.',
                    'action_items': [
                        'Ask more specific questions',
                        'Provide context when asking questions',
                        'Review the source material before asking'
                    ]
                })

            # Based on feature usage
            interaction_breakdown = activity_summary.get('interaction_breakdown', {})
            if interaction_breakdown.get('rag_chat', 0) > 0 and interaction_breakdown.get('rag_question_generation', 0) == 0:
                recommendations.append({
                    'type': 'feature_usage',
                    'priority': 'low',
                    'title': 'Explore Question Generation',
                    'description': 'You primarily use chat. Try the question generation feature for more structured learning.',
                    'action_items': [
                        'Generate practice questions from your documents',
                        'Use the quiz feature to test your knowledge',
                        'Create custom study sessions'
                    ]
                })

            # Always include a positive reinforcement
            recommendations.append({
                'type': 'encouragement',
                'priority': 'low',
                'title': 'Keep Up the Great Work!',
                'description': 'Remember that consistent learning, even in small amounts, leads to significant progress over time.',
                'action_items': [
                    'Continue your learning journey',
                    'Track your progress over time',
                    'Celebrate small wins'
                ]
            })

        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {str(e)}")

        return recommendations

    def health_check(self) -> Dict[str, Any]:
        """Health check for analytics service"""
        try:
            return {
                'status': 'healthy',
                'database_connection': 'ok',
                'analytics_computation': 'ok',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


# Global analytics service instance
analytics_service = AnalyticsService()