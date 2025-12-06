# Use a slim Python image
FROM python:3.10-slim

WORKDIR /app

# Required system dependencies for FAISS and Postgres
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY services/database/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the shared source code
COPY src/ /app/src/
COPY pyproject.toml /app/pyproject.toml
COPY configs /app/configs

# Copy the service code
COPY services/database/main.py /app/main.py

# Set python path to root so imports work
ENV PYTHONPATH=/app
# Default port if not passed by env
ENV DATABASE_PORT=8800

# Run the application
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${DATABASE_PORT}"]