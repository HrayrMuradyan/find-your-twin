# Use a slim Python image
FROM python:3.10-slim

WORKDIR /app

# Install System Dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libpq-dev \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy Project Definition & Source Code
COPY pyproject.toml .
COPY src/ src/

# Install the Package with [database] dependencies
RUN pip install --no-cache-dir ".[database]"

COPY configs configs
COPY services/database/main.py /app/main.py

ENV DATABASE_PORT=8800

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${DATABASE_PORT}"]