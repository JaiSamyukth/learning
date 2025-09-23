"""
RAG-Enhanced Routes
Additional routes for RAG functionality that complement existing routes
"""

from fastapi import APIRouter, HTTPException, Request, Query
from typing import Optional, Dict, Any
from pydantic import BaseModel
import hashlib

from services.rag_integration_service import rag_integration_service
from utils.logger import get_logger

router = APIRouter(prefix="/api/rag", tags=["rag"])
rag_logger = get_logger("rag_routes")

def get_simple_user_id(request: Request) -> str:
    """
    Create a simple user ID based on client IP to isolate users.
    Same function as in other route files for consistency.
    """
    client_ip = request.client.host if request.client else "unknown"
    # Create a simple hash of the IP for session isolation
    user_id = hashlib.md5(client_ip.encode()).hexdigest()[:12]
    return f"user_{user_id}"

class RAGChatRequest(BaseModel):
    message: str

class RAGQuestionRequest(BaseModel):
    topic: Optional[str] = None
    count: int = 25
    mode: str = "practice"

@router.get("/status")
async def get_rag_status(request: Request):
    """Get RAG system status for the current user"""
    try:
        user_token = get_simple_user_id(request)
        status = await rag_integration_service.get_rag_status_for_user(user_token)
        return {
            "success": True,
            "data": status
        }
    except Exception as e:
        rag_logger.error(f"Failed to get RAG status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get RAG status: {str(e)}")

@router.post("/chat")
async def rag_enhanced_chat(request: RAGChatRequest, http_request: Request):
    """RAG-enhanced chat endpoint"""
    try:
        user_token = get_simple_user_id(http_request)
        result = await rag_integration_service.enhanced_chat(request.message, user_token)
        
        return {
            "success": True,
            "data": result
        }
    except Exception as e:
        rag_logger.error(f"RAG chat failed: {e}")
        raise HTTPException(status_code=500, detail=f"RAG chat failed: {str(e)}")

@router.post("/questions")
async def rag_enhanced_questions(request: RAGQuestionRequest, http_request: Request):
    """RAG-enhanced question generation endpoint"""
    try:
        user_token = get_simple_user_id(http_request)
        result = await rag_integration_service.enhanced_question_generation(
            user_token, request.topic, request.count, request.mode
        )
        
        return {
            "success": True,
            "data": result
        }
    except Exception as e:
        rag_logger.error(f"RAG question generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"RAG question generation failed: {str(e)}")

@router.get("/health")
async def rag_health_check():
    """RAG system health check"""
    try:
        health = rag_integration_service.health_check()
        return {
            "success": True,
            "data": health
        }
    except Exception as e:
        rag_logger.error(f"RAG health check failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "data": {
                "status": "unhealthy",
                "error": str(e)
            }
        }

@router.get("/info")
async def rag_system_info():
    """Get RAG system information"""
    try:
        from services.rag_integration_service import RAG_INTEGRATION_AVAILABLE
        
        info = {
            "rag_available": RAG_INTEGRATION_AVAILABLE,
            "integration_version": "1.0.0",
            "features": {
                "semantic_search": RAG_INTEGRATION_AVAILABLE,
                "document_chunking": RAG_INTEGRATION_AVAILABLE,
                "vector_storage": RAG_INTEGRATION_AVAILABLE,
                "enhanced_chat": RAG_INTEGRATION_AVAILABLE,
                "enhanced_questions": RAG_INTEGRATION_AVAILABLE
            }
        }
        
        if RAG_INTEGRATION_AVAILABLE:
            health = rag_integration_service.health_check()
            info["system_health"] = health
        
        return {
            "success": True,
            "data": info
        }
    except Exception as e:
        rag_logger.error(f"Failed to get RAG info: {e}")
        return {
            "success": False,
            "error": str(e)
        }
