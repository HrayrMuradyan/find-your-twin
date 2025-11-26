# check=skip=JSONArgsRecommended
# Use Python 3.10
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements_docker.txt .
RUN pip install --no-cache-dir -r requirements_docker.txt

# Install system dependencies for OpenCV 
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy all your files into the container
COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
COPY models /app/models
COPY embeddings_store /app/embeddings_store
COPY configs /app/configs
COPY app /app/app

# Create a writable directory for temporary files (for deepface)
RUN mkdir -p /app/.deepface && chmod -R 777 /app/.deepface
ENV DEEPFACE_HOME="/app/.deepface"

# Expose the Hugging Face port
EXPOSE 7860

# Start the app
CMD mkdir -p credentials && \
    if [ -n "$GOOGLE_CLIENT_SECRET" ]; then echo "$GOOGLE_CLIENT_SECRET" > credentials/client_secret.json; fi && \
    if [ -n "$GOOGLE_TOKEN_JSON" ]; then echo "$GOOGLE_TOKEN_JSON" > credentials/token.json; fi && \
    uvicorn app.main:app --host 0.0.0.0 --port 7860