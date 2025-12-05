# Use a slim Python image
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
# libgl1 and libglib2.0-0 are required for OpenCV (cv2)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY services/inference/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install scrfd --no-deps

# Copy the shared source code
COPY src/ /app/src/
COPY pyproject.toml /app/pyproject.toml
COPY configs/ /app/configs/
COPY models/ /app/models/

ENV DEEPFACE_HOME=/app/models/embeddings/deepface

# Copy the service code
COPY services/inference/main.py /app/main.py

# Set paths
ENV PYTHONPATH=/app
ENV PROJECT_ROOT=/app

# Run the application
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${INFERENCE_PORT}"]