FROM python:3.10-slim

# Install system dependencies needed by OpenCV and TensorFlow
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project
COPY . .

# Create necessary directories
RUN mkdir -p static/uploads models

# Hugging Face Spaces runs as non-root user 1000
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Start gunicorn
CMD ["gunicorn", "app:app", \
     "--workers", "1", \
     "--timeout", "180", \
     "--bind", "0.0.0.0:7860", \
     "--log-level", "info"]
