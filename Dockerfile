FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*
    
# Create cache directory with proper permissions
RUN mkdir -p /.cache && chmod 777 /.cache

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY api.py .
COPY static/ static/

# Static files are already copied in the previous step
# No need to copy frontend build files separately

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=7860

# Expose the port Hugging Face Spaces expects
EXPOSE 7860

# Command to run the application
CMD ["python", "api.py"]
