# Gunicorn Configuration for Production
import multiprocessing
import os

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
timeout = 30
keepalive = 2

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Logging
loglevel = "info"
accesslog = "/app/backend/logs/access.log"
errorlog = "/app/backend/logs/error.log"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'lumina_iq_rag'

# Server mechanics
preload_app = True
pidfile = "/app/backend/logs/gunicorn.pid"
user = "lumina_iq"
group = "lumina_iq"
tmp_upload_dir = None

# Application
wsgi_module = "main_rag:app"

# Worker timeout
graceful_timeout = 30

# Restart workers after this many requests
max_requests = 1000
max_requests_jitter = 50

# Environment
raw_env = [
    f"QDRANT_URL={os.getenv('QDRANT_URL', 'https://1f6b3bbc-d09e-40c2-b333-0a823825f876.europe-west3-0.gcp.cloud.qdrant.io:6333')}",
    f"QDRANT_API_KEY={os.getenv('QDRANT_API_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.wnnpEmwXoHOjNJ1CTdGaFgqoG7zgLO3O-8bhUbPmK_o')}",
]