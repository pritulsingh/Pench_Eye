FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV + psycopg2 + libmagic
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libpq-dev \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set Python path to support app and ml packages
ENV PYTHONPATH=/app:/app/ml

# Create storage directories
RUN mkdir -p /app/storage/active /app/storage/quarantine

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
