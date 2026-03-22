FROM python:3.12-slim-bookworm

WORKDIR /app

# Environment optimizations
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Copy requirements first (better caching)
COPY requirements.txt .

# Install CPU-only torch first so sentence-transformers does not pull CUDA wheels
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple torch && \
	pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY ./app ./app

# (Optional but cleaner)
EXPOSE 10000

# 🚀 CRITICAL FIX: use $PORT instead of 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1"]
