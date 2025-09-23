# Production Dockerfile for Lumina IQ RAG System
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libtiff5-dev \
    libopenjp2-7-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash lumina_iq
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements_rag.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_rag.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/backend/database /app/backend/temp /app/backend/logs && \
    chown -R lumina_iq:lumina_iq /app

# Switch to non-root user
USER lumina_iq

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start application with gunicorn
CMD ["gunicorn", "--config", "gunicorn.conf.py", "main_rag:app"]