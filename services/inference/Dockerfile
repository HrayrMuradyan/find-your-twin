# Use a slim Python image
FROM python:3.10-slim

WORKDIR /app

# Install System Dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy Project Definition & Source Code
COPY pyproject.toml .
COPY src/ src/

# Install the Package with [inference] dependencies
RUN pip install --no-cache-dir ".[inference]" && \
    pip install scrfd --no-deps

COPY configs/ configs/
COPY models/ models/
COPY services/inference/main.py /app/main.py

# Environment Config
ENV DEEPFACE_HOME=/app/models/embeddings/deepface
ENV INFERENCE_PORT=7860

# Run Application
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${INFERENCE_PORT}"]