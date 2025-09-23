from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import warning suppression first
from utils.suppress_warnings import suppress_third_party_warnings

from config.settings import settings
from routes import auth, pdf, chat
# RAG routes temporarily disabled for testing
# from routes import rag
from utils.logging_config import configure_logging, get_logger
import asyncio
import platform
import logging
import os

# Windows-compatible async optimizations
if platform.system() == "Windows":
    # Use ProactorEventLoop for better Windows performance
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
else:
    # Try uvloop on Unix systems
    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    # Configure enhanced logging
    worker_id = configure_logging()
    logger = get_logger("main")

    if worker_id == '1':  # Only log from first worker
        logger.info("Starting Learning App API...")
    yield
    # Shutdown
    if worker_id == '1':  # Only log from first worker
        print("Shutting down Learning App API...")

app = FastAPI(
    title="Learning App API", 
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(pdf.router)
app.include_router(chat.router)
# RAG router temporarily disabled for testing
# app.include_router(rag.router)

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "Learning App API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=settings.HOST, 
        port=settings.PORT,
        workers=1,  # Use 1 worker for development, increase for production
        loop="asyncio",  # Use asyncio (Windows compatible)
        access_log=True,
        log_level="info",
        # Windows-compatible performance settings
        backlog=2048,
        timeout_keep_alive=5
    )
