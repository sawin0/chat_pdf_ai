FROM python:3.12-slim

WORKDIR /app

# Environment optimizations
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY ./app ./app

# (Optional but cleaner)
EXPOSE 10000

# 🚀 CRITICAL FIX: use $PORT instead of 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1"]
