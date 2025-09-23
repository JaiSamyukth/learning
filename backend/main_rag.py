"""
Lumina IQ RAG-Enhanced Main Application
Production-grade FastAPI application with comprehensive RAG capabilities
"""

import asyncio
import logging
from typing import Optional
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn

from backend.config.qdrant_config import qdrant_config
from backend.services.database_service import database_service
from backend.services.vector_service import vector_service
from backend.services.rag_service import rag_service
from backend.services.rag_chat_service import rag_chat_service
from backend.services.user_management_service import user_management_service
from backend.services.rag_config_service import rag_config_service
from backend.services.analytics_service import analytics_service
from backend.utils.logger import setup_logger


# FastAPI application
app = FastAPI(
    title="Lumina IQ RAG-Enhanced API",
    description="Production-grade AI-powered learning assistant with RAG capabilities",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Security
security = HTTPBearer(auto_error=False)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for all unhandled errors"""
    logging.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
            "timestamp": datetime.now().isoformat()
        }
    )


# Health check endpoint
@app.get("/health")
async def health_check():
    """Comprehensive health check for all services"""
    try:
        # Check all service health
        db_health = database_service.health_check()
        vector_health = vector_service.health_check()
        rag_health = rag_service.health_check()
        chat_health = rag_chat_service.health_check()
        user_health = user_management_service.health_check()
        config_health = rag_config_service.health_check()
        analytics_health = analytics_service.health_check()

        # Overall health status
        service_statuses = [
            db_health.get('status'),
            vector_health.get('status'),
            rag_health.get('status'),
            chat_health.get('status'),
            user_health.get('status'),
            config_health.get('status'),
            analytics_health.get('status')
        ]

        overall_status = "healthy" if all(s == "healthy" for s in service_statuses) else "degraded"

        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "services": {
                "database": db_health,
                "vector": vector_health,
                "rag": rag_health,
                "chat": chat_health,
                "user_management": user_health,
                "config": config_health,
                "analytics": analytics_health
            },
            "version": "2.0.0"
        }
    except Exception as e:
        logging.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# User Management Endpoints
class UserCreateRequest(BaseModel):
    username: str
    email: str
    password: str


class UserLoginRequest(BaseModel):
    username: str
    password: str


class UserUpdateRequest(BaseModel):
    email: Optional[str] = None
    rag_enabled: Optional[bool] = None
    embedding_quota: Optional[int] = None


@app.post("/auth/register")
async def register_user(request: UserCreateRequest):
    """Register a new user"""
    try:
        result = await user_management_service.create_user(request)
        return {"message": "User created successfully", "data": result}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"User registration failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Registration failed")


@app.post("/auth/login")
async def login_user(request: UserLoginRequest):
    """Login user"""
    try:
        result = await user_management_service.login_user(request)
        return {"message": "Login successful", "data": result}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"User login failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Login failed")


@app.get("/auth/profile")
async def get_user_profile(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get user profile"""
    try:
        if not credentials:
            raise HTTPException(status_code=401, detail="Authentication required")

        result = await user_management_service.get_user_profile(credentials.credentials)
        return {"message": "Profile retrieved successfully", "data": result}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Profile retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve profile")


@app.put("/auth/profile")
async def update_user_profile(
    request: UserUpdateRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Update user profile"""
    try:
        if not credentials:
            raise HTTPException(status_code=401, detail="Authentication required")

        result = await user_management_service.update_user_profile(credentials.credentials, request)
        return {"message": "Profile updated successfully", "data": result}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Profile update failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update profile")


@app.post("/auth/logout")
async def logout_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Logout user"""
    try:
        if not credentials:
            raise HTTPException(status_code=401, detail="Authentication required")

        result = await user_management_service.logout_user(credentials.credentials)
        return {"message": "Logged out successfully", "data": result}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Logout failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Logout failed")


# RAG Configuration Endpoints
class RAGConfigRequest(BaseModel):
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    embedding_model: Optional[str] = None
    retrieval_top_k: Optional[int] = None
    rerank_enabled: Optional[bool] = None
    search_score_threshold: Optional[float] = None


@app.get("/rag/config")
async def get_rag_config(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get RAG configuration for user"""
    try:
        if not credentials:
            raise HTTPException(status_code=401, detail="Authentication required")

        user_id = user_management_service._get_user_id_from_session(credentials.credentials)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid session")

        result = rag_config_service.get_user_config(user_id)
        return {"message": "Configuration retrieved successfully", "data": result}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Config retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve configuration")


@app.put("/rag/config")
async def update_rag_config(
    request: RAGConfigRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Update RAG configuration for user"""
    try:
        if not credentials:
            raise HTTPException(status_code=401, detail="Authentication required")

        user_id = user_management_service._get_user_id_from_session(credentials.credentials)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid session")

        result = rag_config_service.update_user_config(user_id, request.dict(exclude_unset=True))
        return {"message": "Configuration updated successfully", "data": result}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Config update failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update configuration")


@app.get("/rag/config/presets")
async def get_rag_presets():
    """Get RAG configuration presets"""
    try:
        result = rag_config_service.get_config_presets()
        return {"message": "Presets retrieved successfully", "data": result}
    except Exception as e:
        logging.error(f"Preset retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve presets")


# Document Processing Endpoints
@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Upload and process a document with RAG"""
    try:
        if not credentials:
            raise HTTPException(status_code=401, detail="Authentication required")

        user_id = user_management_service._get_user_id_from_session(credentials.credentials)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid session")

        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        allowed_extensions = {'.pdf', '.docx', '.txt', '.epub', '.html'}
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_extension}")

        # Save file temporarily
        temp_path = Path(f"backend/temp/{file.filename}")
        temp_path.parent.mkdir(parents=True, exist_ok=True)

        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Process document with RAG
        result = await rag_service.process_document(str(temp_path), file.filename, str(user_id))

        # Clean up temp file
        temp_path.unlink(missing_ok=True)

        return {
            "message": "Document processed successfully with RAG",
            "data": result
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Document upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")


@app.get("/documents")
async def get_user_documents(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get user's documents"""
    try:
        if not credentials:
            raise HTTPException(status_code=401, detail="Authentication required")

        user_id = user_management_service._get_user_id_from_session(credentials.credentials)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid session")

        documents = database_service.get_user_documents(user_id)
        return {"message": "Documents retrieved successfully", "data": documents}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Document retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")


# Enhanced Chat Endpoints
@app.post("/chat/rag")
async def rag_chat(
    request: dict,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """RAG-enhanced chat"""
    try:
        if not credentials:
            raise HTTPException(status_code=401, detail="Authentication required")

        message = request.get("message", "")
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")

        result = await rag_chat_service.chat(message, credentials.credentials, use_rag=True)
        return {"message": "RAG chat completed", "data": result}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"RAG chat failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.get("/chat/history")
async def get_chat_history(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get combined chat history"""
    try:
        if not credentials:
            raise HTTPException(status_code=401, detail="Authentication required")

        result = await rag_chat_service.get_chat_history(credentials.credentials)
        return {"message": "Chat history retrieved successfully", "data": result}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Chat history retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history")


@app.post("/chat/clear")
async def clear_chat_history(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Clear chat history"""
    try:
        if not credentials:
            raise HTTPException(status_code=401, detail="Authentication required")

        result = await rag_chat_service.clear_chat_history(credentials.credentials)
        return {"message": "Chat history cleared successfully", "data": result}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Chat history clearing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to clear chat history")


# Analytics Endpoints
@app.get("/analytics/user")
async def get_user_analytics(
    days: int = 30,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get user analytics"""
    try:
        if not credentials:
            raise HTTPException(status_code=401, detail="Authentication required")

        user_id = user_management_service._get_user_id_from_session(credentials.credentials)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid session")

        result = await analytics_service.get_user_activity_summary(user_id, days)
        return {"message": "User analytics retrieved successfully", "data": result}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"User analytics retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve user analytics")


@app.get("/analytics/system")
async def get_system_analytics(days: int = 7):
    """Get system-wide analytics"""
    try:
        result = await analytics_service.get_system_analytics(days)
        return {"message": "System analytics retrieved successfully", "data": result}
    except Exception as e:
        logging.error(f"System analytics retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system analytics")


@app.get("/analytics/progress")
async def get_user_progress(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get user progress report"""
    try:
        if not credentials:
            raise HTTPException(status_code=401, detail="Authentication required")

        user_id = user_management_service._get_user_id_from_session(credentials.credentials)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid session")

        result = await analytics_service.get_user_progress_report(user_id, days=30)
        return {"message": "Progress report generated successfully", "data": result}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Progress report generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate progress report")


# System Status Endpoints
@app.get("/status/rag")
async def get_rag_status(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get RAG system status for user"""
    try:
        if not credentials:
            raise HTTPException(status_code=401, detail="Authentication required")

        result = await rag_chat_service.get_rag_status(credentials.credentials)
        return {"message": "RAG status retrieved successfully", "data": result}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"RAG status retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve RAG status")


@app.get("/status/quota")
async def get_user_quota(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get user quota status"""
    try:
        if not credentials:
            raise HTTPException(status_code=401, detail="Authentication required")

        user_id = user_management_service._get_user_id_from_session(credentials.credentials)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid session")

        profile = await user_management_service.get_user_profile(credentials.credentials)
        return {
            "message": "Quota status retrieved successfully",
            "data": {
                "quota_total": profile['data']['embedding_quota'],
                "quota_used": profile['data']['quota_used'],
                "quota_remaining": profile['data']['quota_remaining']
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Quota status retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve quota status")


# Utility Endpoints
@app.post("/maintenance/cleanup")
async def cleanup_old_data(days_to_keep: int = 90):
    """Clean up old data (admin endpoint)"""
    try:
        result = database_service.cleanup_old_data(days_to_keep)
        return {"message": "Cleanup completed successfully", "data": result}
    except Exception as e:
        logging.error(f"Cleanup failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Cleanup failed")


@app.post("/test/qdrant-connection")
async def test_qdrant_connection():
    """Test Qdrant Cloud connection"""
    try:
        health = vector_service.health_check()
        return {"message": "Qdrant connection test completed", "data": health}
    except Exception as e:
        logging.error(f"Qdrant connection test failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Qdrant connection test failed")


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Application startup"""
    logging.info("Starting Lumina IQ RAG-Enhanced API v2.0.0")

    # Test all service connections
    try:
        # Test database
        db_health = database_service.health_check()
        logging.info(f"Database health: {db_health.get('status', 'unknown')}")

        # Test vector service
        vector_health = vector_service.health_check()
        logging.info(f"Vector service health: {vector_health.get('status', 'unknown')}")

        # Test RAG service
        rag_health = rag_service.health_check()
        logging.info(f"RAG service health: {rag_health.get('status', 'unknown')}")

        logging.info("All services initialized successfully")

    except Exception as e:
        logging.error(f"Service initialization failed: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    logging.info("Shutting down Lumina IQ RAG-Enhanced API")

    # Cleanup operations
    try:
        # Clean up temporary files
        temp_dir = Path("backend/temp")
        if temp_dir.exists():
            for file in temp_dir.glob("*"):
                file.unlink(missing_ok=True)
            temp_dir.rmdir()

        logging.info("Cleanup completed successfully")
    except Exception as e:
        logging.error(f"Cleanup failed: {str(e)}")


# Error handlers for specific exception types
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with proper logging"""
    logging.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "Request failed",
            "message": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    # Production server configuration
    uvicorn.run(
        "main_rag:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        log_level="info",
        access_log=True,
        server_header=False,
        date_header=False
    )