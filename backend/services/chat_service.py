# Import warning suppression first
from utils.suppress_warnings import suppress_third_party_warnings

import google.generativeai as genai
from datetime import datetime
from fastapi import HTTPException
from config.settings import settings
from utils.storage import pdf_contexts, chat_histories, storage_manager
from models.chat import (
    ChatMessage,
    ChatResponse,
    AnswerEvaluationRequest,
    AnswerEvaluationResponse,
    QuizSubmissionRequest,
    QuizSubmissionResponse,
    QuizAnswer
)
import sys
import os
import logging
import asyncio
import concurrent.futures
from threading import Lock
from utils.logger import get_logger

# Use enhanced logger
chat_logger = get_logger("chat_service")

# RAG Integration - Lazy import to avoid circular imports
rag_integration_service = None

def get_rag_integration_service():
    global rag_integration_service
    if rag_integration_service is None:
        try:
            from services.rag_integration_service import rag_integration_service as ris
            rag_integration_service = ris
            chat_logger.info("RAG integration loaded for chat service")
        except ImportError as e:
            chat_logger.warning(f"RAG integration not available: {e}")
    return rag_integration_service

# Add the parent directory to the path to import from api_rotation
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from api_rotation.api_key_rotator import api_key_rotator
    ROTATION_ENABLED = True
    # Only log once per module import to avoid duplicate logs with multiple workers
    if not hasattr(chat_logger, '_rotation_logged'):
        chat_logger.info("API key rotation enabled")
        chat_logger._rotation_logged = True
except ImportError as e:
    if not hasattr(chat_logger, '_rotation_error_logged'):
        chat_logger.warning(f"API key rotation not available: {e}")
        chat_logger._rotation_error_logged = True
    ROTATION_ENABLED = False

# Thread-safe locks for concurrent access
from collections import defaultdict
user_locks = defaultdict(Lock)  # Per-user locks for better concurrency
model_cache = {}
model_cache_lock = Lock()

# Massive thread pool for true concurrency with 25+ users
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=100)

# Massive thread pools optimized for 3000+ concurrent requests with 14 API keys
ai_generation_pool = concurrent.futures.ThreadPoolExecutor(max_workers=500)  # Massive pool for 3000+ users
model_creation_pool = concurrent.futures.ThreadPoolExecutor(max_workers=200)  # Large pool for model creation

# Request queue optimized for 3000+ concurrent requests
import queue
request_queue = asyncio.Queue(maxsize=5000)  # Large queue for massive burst traffic
request_semaphore = asyncio.Semaphore(20)  # Very conservative to prevent quota exhaustion

# Performance monitoring
import time
request_times = []
request_times_lock = Lock()

# Optimized rate limiting for 14 API keys handling 3000+ requests
from collections import defaultdict
api_call_times = defaultdict(list)  # Track API calls per key
rate_limit_lock = Lock()
CALLS_PER_MINUTE = 1800  # 30 req/sec * 60 sec = 1800 per minute per API key
QUOTA_RESET_TIME = 60  # Reset quota tracking every minute
# Total capacity: 14 keys * 1800 = 25,200 requests/minute = 420 requests/second!

# Connection pooling for AI requests
import aiohttp
ai_session = None
ai_session_lock = asyncio.Lock()

# Global rate limiter to prevent overwhelming the API
last_request_time = 0
min_request_interval = 0.05  # Minimum 50ms between requests globally

def check_rate_limit(api_key: str) -> bool:
    """Check if API key is within rate limits"""
    current_time = time.time()

    with rate_limit_lock:
        # Clean old entries
        api_call_times[api_key] = [
            call_time for call_time in api_call_times[api_key]
            if current_time - call_time < QUOTA_RESET_TIME
        ]

        # Check if under limit
        return len(api_call_times[api_key]) < CALLS_PER_MINUTE

def record_api_call(api_key: str):
    """Record an API call for rate limiting"""
    current_time = time.time()

    with rate_limit_lock:
        api_call_times[api_key].append(current_time)

async def get_rotated_model_async():
    """
    Get a Gemini model instance optimized for 3000+ concurrent requests.
    Uses 14 API keys with intelligent load balancing.
    """
    loop = asyncio.get_event_loop()

    # Get API key with optimized rotation for high throughput
    def get_high_throughput_api_key():
        if ROTATION_ENABLED:
            # With 14 API keys, we can be more aggressive
            # Try a few keys to find one that's not heavily loaded
            for attempt in range(3):  # Only try 3 times for speed
                api_key = api_key_rotator.get_next_key()
                if api_key and check_rate_limit(api_key):
                    return api_key

            # If all checked keys are busy, just use the next one
            # With 14 keys, the load should be well distributed
            api_key = api_key_rotator.get_next_key()
            if api_key:
                return api_key

            chat_logger.warning("API rotation failed, using settings key")

        return settings.GEMINI_API_KEY

    api_key = await loop.run_in_executor(model_creation_pool, get_high_throughput_api_key)

    if not api_key:
        raise HTTPException(status_code=500, detail="No API key available")

    # Record the API call for monitoring (but don't block on it)
    record_api_call(api_key)

    # Create model optimized for high concurrency
    def create_model():
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(settings.GEMINI_MODEL)

    return await loop.run_in_executor(model_creation_pool, create_model)

async def generate_content_async(model, context: str, max_retries: int = 2, priority: str = "normal") -> str:
    """
    Generate content asynchronously optimized for 25+ concurrent users.
    Uses semaphore-based throttling and dedicated thread pools.
    """
    start_time = time.time()
    loop = asyncio.get_event_loop()

    # Use semaphore to limit concurrent AI requests and prevent overload
    async with request_semaphore:
        # Global rate limiting to prevent overwhelming API
        global last_request_time
        current_time = time.time()
        time_since_last = current_time - last_request_time
        if time_since_last < min_request_interval:
            await asyncio.sleep(min_request_interval - time_since_last)
        last_request_time = time.time()

        for attempt in range(max_retries):
            try:
                # Add larger delay to spread requests across time and prevent quota exhaustion
                await asyncio.sleep(0.1 + (hash(str(model)) % 100) / 1000)  # 0.1-0.2s spread

                # Use dedicated AI generation pool with throttling
                response = await loop.run_in_executor(
                    ai_generation_pool,
                    lambda: model.generate_content(context)
                )
                if response and response.text:
                    # Record performance metrics
                    end_time = time.time()
                    response_time = end_time - start_time

                    with request_times_lock:
                        request_times.append(response_time)
                        # Keep only last 100 requests for monitoring
                        if len(request_times) > 100:
                            request_times.pop(0)

                    chat_logger.debug(f"AI response generated in {response_time:.2f}s (priority: {priority})")
                    return response.text.strip()
                else:
                    raise Exception("Empty response from AI model")

            except Exception as e:
                error_str = str(e)
                # Check if it's a rate limit or quota error
                if any(keyword in error_str.lower() for keyword in ['rate limit', 'quota', 'exhausted', '429']):
                    chat_logger.warning(f"Rate limit/quota hit, waiting before retry {attempt + 1}: {error_str}")
                    # Exponential backoff for rate limits
                    wait_time = min(2.0 ** attempt, 10.0)  # Cap at 10 seconds
                    await asyncio.sleep(wait_time)
                else:
                    chat_logger.warning(f"AI generation attempt {attempt + 1} failed: {error_str}")
                    if attempt == max_retries - 1:
                        chat_logger.error(f"All {max_retries} AI generation attempts failed: {error_str}")
                        raise
                    # Shorter retry delay for other errors
                    await asyncio.sleep(0.3 * (attempt + 1))

    raise Exception("Failed to generate AI response after all retries")

def safe_storage_access(operation, token: str, *args, **kwargs):
    """
    Thread-safe wrapper for storage operations using per-user locks.
    This allows multiple users to access their data concurrently.
    Uses the optimized storage manager for better performance.
    """
    with storage_manager.get_user_lock(token):
        return operation(*args, **kwargs)

class ChatService:
    @staticmethod
    async def chat(message: ChatMessage, token: str) -> ChatResponse:
        """RAG-enhanced chat - no legacy fallback"""
        chat_logger.info("Processing chat message with RAG",
                        token=token,
                        message_length=len(message.message))

        try:
            chat_logger.info("Using RAG-enhanced chat")
            rag_service = get_rag_integration_service()
            if rag_service is None:
                raise Exception("RAG integration service not available")
            rag_result = await rag_service.enhanced_chat(message.message, token)

            if rag_result.get("rag_used", False):
                chat_logger.info("RAG chat successful")
                return ChatResponse(
                    response=rag_result.get("response", ""),
                    timestamp=rag_result.get("timestamp", datetime.now().isoformat())
                )
            else:
                # If RAG is not used, return an error message
                chat_logger.error("RAG system failed to process the request")
                return ChatResponse(
                    response="I'm sorry, but I'm unable to process your request right now. Please ensure you have uploaded a document and try again.",
                    timestamp=datetime.now().isoformat()
                )
        except Exception as e:
            chat_logger.error(f"RAG chat failed: {e}")
            return ChatResponse(
                response="I'm experiencing technical difficulties. Please try again later.",
                timestamp=datetime.now().isoformat()
            )



    @staticmethod
    def get_chat_history(token: str) -> dict:
        def get_history():
            if token not in chat_histories:
                return {"history": []}
            return {"history": chat_histories[token]}

        return safe_storage_access(get_history, token)

    @staticmethod
    def clear_chat_history(token: str) -> dict:
        def clear_history():
            if token in chat_histories:
                chat_histories[token] = []
            return {"message": "Chat history cleared"}

        return safe_storage_access(clear_history, token)

    @staticmethod
    async def generate_questions(token: str, topic: str = None, count: int = 25, mode: str = "practice") -> ChatResponse:
        chat_logger.info("Generating questions with RAG",
                        token=token,
                        topic=topic,
                        count=count,
                        mode=mode)

        try:
            chat_logger.info("Using RAG-enhanced question generation")
            rag_service = get_rag_integration_service()
            if rag_service is None:
                raise Exception("RAG integration service not available")
            rag_result = await rag_service.enhanced_question_generation(token, topic, count, mode)

            if rag_result.get("rag_used", False):
                chat_logger.info("RAG question generation successful")
                import json
                return ChatResponse(
                    response=json.dumps(rag_result.get("questions", [])),
                    timestamp=datetime.now().isoformat()
                )
            else:
                # If RAG is not used, return an error message
                chat_logger.error("RAG system failed to generate questions")
                return ChatResponse(
                    response=json.dumps([{
                        "error": "Unable to generate questions. Please ensure you have uploaded a document and try again."
                    }]),
                    timestamp=datetime.now().isoformat()
                )
        except Exception as e:
            chat_logger.error(f"RAG question generation failed: {e}")
            return ChatResponse(
                response=json.dumps([{
                    "error": "I'm experiencing technical difficulties generating questions. Please try again later."
                }]),
                timestamp=datetime.now().isoformat()
            )


    @staticmethod
    async def evaluate_answer(request: AnswerEvaluationRequest, token: str) -> AnswerEvaluationResponse:
        """Evaluate a single user answer using AI"""
        def get_pdf_context():
            if token not in pdf_contexts:
                raise HTTPException(status_code=400, detail="No PDF selected. Please select a PDF first.")
            return pdf_contexts[token]

        pdf_context = safe_storage_access(get_pdf_context, token)

        # Get evaluation level settings
        evaluation_level = request.evaluation_level or "medium"

        # Define evaluation criteria based on level
        if evaluation_level == "easy":
            criteria_text = """
        EVALUATION LEVEL: EASY (Lenient)
        - Focus on basic understanding and effort
        - Give credit for partial answers and good attempts
        - Be encouraging and supportive in feedback

        SCORING SCALE (0-10):
        - 8-10: Shows basic understanding, good effort
        - 6-7: Partially correct, some understanding shown
        - 4-5: Minimal understanding but attempted
        - 2-3: Little understanding but some effort
        - 0-1: No answer or completely off-topic"""
        elif evaluation_level == "strict":
            criteria_text = """
        EVALUATION LEVEL: STRICT (Rigorous)
        - Require precise, detailed, and comprehensive answers
        - Expect specific examples and thorough explanations
        - Be critical of incomplete or vague responses

        SCORING SCALE (0-10):
        - 9-10: Exceptional - Precise, comprehensive, with specific examples
        - 7-8: Very Good - Accurate and detailed, minor gaps acceptable
        - 5-6: Adequate - Correct but lacks depth or detail
        - 3-4: Below Standard - Significant gaps or inaccuracies
        - 1-2: Poor - Major errors or very incomplete
        - 0: No answer or completely wrong"""
        else:  # medium
            criteria_text = """
        EVALUATION LEVEL: MEDIUM (Balanced)
        - Expect reasonable understanding and adequate detail
        - Balance between being supportive and maintaining standards
        - Look for key concepts and main points

        SCORING SCALE (0-10):
        - 9-10: Excellent - Accurate, complete, demonstrates deep understanding
        - 7-8: Good - Mostly accurate, covers main points, good understanding
        - 5-6: Satisfactory - Partially correct, basic understanding shown
        - 3-4: Needs Improvement - Some correct elements but significant gaps
        - 1-2: Poor - Mostly incorrect or irrelevant
        - 0: No answer provided or completely wrong

        IMPORTANT: If the student's answer is empty, blank, or just says "No answer provided", give a score of 0."""

        # Check if answer is empty and give 0 score
        if not request.user_answer or request.user_answer.strip() == "" or request.user_answer.strip().lower() == "no answer provided":
            return AnswerEvaluationResponse(
                question_id=request.question_id,
                score=0,
                max_score=10,
                feedback="No answer was provided for this question.",
                suggestions="Please provide an answer based on the document content to receive a score."
            )

        # This logic will be handled in the quiz evaluation method instead
        # Individual answer evaluation continues with the original logic for open-ended questions

        # Prepare context for evaluation
        evaluation_context = f"""
        You are an expert educational evaluator. Your task is to evaluate a student's answer to a question based on the provided document content.

        Document: {pdf_context['filename']}
        Document Content: {pdf_context['content'][:20000]}

        Question: {request.question}
        Student's Answer: {request.user_answer}

        EVALUATION CRITERIA:
        1. Accuracy: How correct is the answer based on the document content?
        2. Completeness: Does the answer cover all important aspects?
        3. Understanding: Does the student demonstrate clear understanding?
        4. Relevance: Is the answer relevant to the question asked?

        {criteria_text}

        Please provide your evaluation in the following JSON format:
        {{
            "score": [0-10 integer],
            "feedback": "[Detailed feedback explaining the score, highlighting what was correct and what was missing]",
            "suggestions": "[Specific suggestions for improvement, including where to focus, what has been missed and how to correct]",
            "correct_answer_hint": "[Brief hint about the correct answer without giving it away completely]"
        }}

        Be constructive and encouraging in your feedback while being honest about areas for improvement.
        """

        try:
            rotated_model = await get_rotated_model_async()
            ai_response = await generate_content_async(rotated_model, evaluation_context)

            # Try to extract JSON from the response
            import json
            import re

            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', ai_response)
            if json_match:
                try:
                    evaluation_data = json.loads(json_match.group())

                    return AnswerEvaluationResponse(
                        question_id=request.question_id,
                        score=min(max(evaluation_data.get('score', 0), 0), 10),  # Ensure score is 0-10
                        feedback=evaluation_data.get('feedback', 'No feedback provided'),
                        suggestions=evaluation_data.get('suggestions', 'No suggestions provided'),
                        correct_answer_hint=evaluation_data.get('correct_answer_hint')
                    )
                except json.JSONDecodeError:
                    pass

            # Fallback if JSON parsing fails
            return AnswerEvaluationResponse(
                question_id=request.question_id,
                score=5,  # Default middle score
                feedback=ai_response,
                suggestions="Please review the document content for more accurate information.",
                correct_answer_hint="Refer to the relevant sections in the document."
            )

        except Exception as e:
            chat_logger.error("Error evaluating answer", error=str(e))
            raise HTTPException(status_code=500, detail=f"Failed to evaluate answer: {str(e)}")

    @staticmethod
    async def evaluate_quiz(request: QuizSubmissionRequest, token: str) -> QuizSubmissionResponse:
        """Evaluate a complete quiz submission"""
        def get_pdf_context():
            if token not in pdf_contexts:
                raise HTTPException(status_code=400, detail="No PDF selected. Please select a PDF first.")
            return pdf_contexts[token]

        pdf_context = safe_storage_access(get_pdf_context, token)

        # Evaluate each answer individually
        individual_results = []
        total_score = 0

        for answer in request.answers:
            # Handle MCQ questions with binary scoring
            if answer.question_type == 'mcq' and answer.correct_answer:
                user_answer_clean = answer.user_answer.strip().upper()
                correct_answer_clean = answer.correct_answer.strip().upper()

                # We need to get the actual answer text from the question to provide meaningful feedback
                # For now, we'll use AI to get the correct answer explanation
                try:
                    # Get the correct answer explanation using AI
                    explanation_context = f"""
                    You are an educational AI providing feedback on a multiple choice question.

                    Document: {pdf_context['filename']}
                    Document Content: {pdf_context['content'][:15000]}

                    Question: {answer.question}
                    Correct Answer: Option {correct_answer_clean}

                    Please provide a brief explanation (1-2 sentences) of why option {correct_answer_clean} is the correct answer based on the document content.
                    Focus on the specific information from the document that supports this answer.

                    Respond with just the explanation, no additional formatting.
                    """

                    rotated_model = await get_rotated_model_async()
                    explanation_response = await generate_content_async(rotated_model, explanation_context)
                    correct_answer_explanation = explanation_response.strip() if explanation_response else f"Option {correct_answer_clean} is the correct answer according to the document."

                except Exception as e:
                    print(f"Error getting answer explanation: {e}")
                    correct_answer_explanation = f"Option {correct_answer_clean} is the correct answer according to the document."

                # Get the actual option texts for better feedback
                user_option_text = ""
                correct_option_text = ""

                if answer.options:
                    # Find the option texts
                    for option in answer.options:
                        if option.startswith(f"{user_answer_clean})"):
                            user_option_text = option
                        if option.startswith(f"{correct_answer_clean})"):
                            correct_option_text = option

                # Binary scoring for MCQ: either full marks or zero
                if user_answer_clean == correct_answer_clean:
                    score = 10  # Full marks for correct answer
                    feedback = f"✅ Correct! You selected '{user_option_text or f'Option {user_answer_clean}'}'. {correct_answer_explanation}"
                    suggestions = "Great job! Continue studying to maintain this level of understanding."
                elif not answer.user_answer or answer.user_answer.strip() == "" or answer.user_answer.strip().lower() == "no answer provided":
                    score = 0  # Zero marks for no answer
                    feedback = f"❌ No answer was provided for this question. The correct answer is '{correct_option_text or f'Option {correct_answer_clean}'}'. {correct_answer_explanation}"
                    suggestions = "Please provide an answer based on the document content to receive a score."
                else:
                    score = 0  # Zero marks for incorrect answer
                    feedback = f"❌ Incorrect. You selected '{user_option_text or f'Option {user_answer_clean}'}', but the correct answer is '{correct_option_text or f'Option {correct_answer_clean}'}'. {correct_answer_explanation}"
                    suggestions = "Review the relevant section in the document to understand the correct answer."

                result = AnswerEvaluationResponse(
                    question_id=answer.question_id,
                    score=score,
                    max_score=10,
                    feedback=feedback,
                    suggestions=suggestions
                )
            else:
                # Handle open-ended questions with AI evaluation
                eval_request = AnswerEvaluationRequest(
                    question=answer.question,
                    user_answer=answer.user_answer,
                    question_id=answer.question_id,
                    evaluation_level=request.evaluation_level
                )
                result = await ChatService.evaluate_answer(eval_request, token)

            individual_results.append(result)
            total_score += result.score

        # Calculate overall metrics
        max_possible_score = len(request.answers) * 10
        percentage = (total_score / max_possible_score) * 100 if max_possible_score > 0 else 0

        # Determine grade
        if percentage >= 90:
            grade = "A"
        elif percentage >= 80:
            grade = "B"
        elif percentage >= 70:
            grade = "C"
        elif percentage >= 60:
            grade = "D"
        else:
            grade = "F"

        # Generate overall feedback using AI
        overall_context = f"""
        You are an educational AI providing comprehensive feedback on a student's quiz performance.

        Document: {pdf_context['filename']}
        Topic: {request.topic or 'General'}

        Quiz Results:
        - Total Score: {total_score}/{max_possible_score} ({percentage:.1f}%)
        - Grade: {grade}
        - Number of Questions: {len(request.answers)}

        Individual Question Performance:
        """

        for i, (answer, result) in enumerate(zip(request.answers, individual_results), 1):
            overall_context += f"""
        Question {i}: {answer.question}
        Student Answer: {answer.user_answer}
        Score: {result.score}/10
        """

        overall_context += f"""

        Based on this performance, provide:
        1. Overall feedback (2-3 sentences about the student's performance)
        2. Study suggestions (3-4 specific recommendations)
        3. Strengths (2-3 areas where the student performed well)
        4. Areas for improvement (2-3 specific areas needing work)

        Provide your response in JSON format:
        {{
            "overall_feedback": "[Overall assessment of performance]",
            "study_suggestions": ["suggestion1", "suggestion2", "suggestion3"],
            "strengths": ["strength1", "strength2"],
            "areas_for_improvement": ["area1", "area2", "area3"]
        }}

        Be encouraging and constructive while providing actionable feedback.
        """

        try:
            rotated_model = await get_rotated_model_async()
            ai_response = await generate_content_async(rotated_model, overall_context)

            # Parse AI response
            import json
            import re

            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', ai_response)
            if json_match:
                try:
                    feedback_data = json.loads(json_match.group())

                    return QuizSubmissionResponse(
                        overall_score=total_score,
                        max_score=max_possible_score,
                        percentage=round(percentage, 1),
                        grade=grade,
                        individual_results=individual_results,
                        overall_feedback=feedback_data.get('overall_feedback', f'You scored {total_score}/{max_possible_score} ({percentage:.1f}%)'),
                        study_suggestions=feedback_data.get('study_suggestions', ['Review the document content', 'Practice more questions']),
                        strengths=feedback_data.get('strengths', ['Attempted all questions']),
                        areas_for_improvement=feedback_data.get('areas_for_improvement', ['Focus on accuracy', 'Provide more detailed answers'])
                    )
                except json.JSONDecodeError:
                    pass

            # Fallback response
            return QuizSubmissionResponse(
                overall_score=total_score,
                max_score=max_possible_score,
                percentage=round(percentage, 1),
                grade=grade,
                individual_results=individual_results,
                overall_feedback=f'You scored {total_score}/{max_possible_score} ({percentage:.1f}%). {"Great job!" if percentage >= 80 else "Keep practicing to improve your understanding."}',
                study_suggestions=['Review the document content thoroughly', 'Focus on key concepts and definitions', 'Practice explaining concepts in your own words'],
                strengths=['Completed all questions', 'Showed effort in answering'],
                areas_for_improvement=['Accuracy of responses', 'Depth of understanding', 'Use of specific examples from the text']
            )

        except Exception as e:
            print(f"Error generating overall feedback: {str(e)}")
            # Return basic response without AI-generated feedback
            return QuizSubmissionResponse(
                overall_score=total_score,
                max_score=max_possible_score,
                percentage=round(percentage, 1),
                grade=grade,
                individual_results=individual_results,
                overall_feedback=f'Quiz completed. Score: {total_score}/{max_possible_score} ({percentage:.1f}%)',
                study_suggestions=['Review the document content', 'Practice more questions'],
                strengths=['Completed the quiz'],
                areas_for_improvement=['Continue studying the material']
            )
