# Use a slim Python image
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
# libgomp1 is often required for FAISS
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
# Ensure requirements.txt includes: faiss-cpu, numpy, psycopg2-binary, fastapi, uvicorn, python-dotenv
COPY services/database/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the shared source code
# IMPORTANT: This assumes your CI/CD pipeline (GitHub Action) copies the 'src' folder
# into the 'services/database' folder before building.
COPY src/ /app/src/
COPY pyproject.toml /app/pyproject.toml
COPY configs /app/configs

# Copy the service code
COPY services/database/main.py /app/main.py

# Set python path to root so imports work
ENV PYTHONPATH=/app

# Run the application
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${DATABASE_PORT}"]