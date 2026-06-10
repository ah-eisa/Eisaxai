FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers (for PDF export)
RUN pip install playwright && python -m playwright install --with-deps chromium

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p /app/file_cache /app/uploads /app/static /app/logs /app/backups

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f -H "X-API-Key: ${SECURE_TOKEN}" http://localhost:8000/health || exit 1

# Run with uvicorn
CMD ["python", "-m", "uvicorn", "api_bridge_v2:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
