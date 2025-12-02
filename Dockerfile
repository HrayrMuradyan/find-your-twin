# Use Python 3.10
FROM python:3.10-slim

# Set the working directory
WORKDIR /root

# Install system dependencies
# libgl1/glib2 are needed for OpenCV (used by deepface/scrfd)
# Supervisor is removed
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements_docker.txt .
RUN pip install --no-cache-dir -r requirements_docker.txt && \
    pip install scrfd --no-deps

# Create project structure
COPY pyproject.toml /root/pyproject.toml
COPY src /root/src
COPY models /root/models
COPY configs /root/configs
COPY app /root/app


# Create credentials folder (will be populated by run.py from env vars)
RUN mkdir -p /root/credentials

# Create writable directory for deepface weights
RUN mkdir -p /root/.deepface && chmod -R 777 /root/.deepface
ENV DEEPFACE_HOME="/root/.deepface"

# Env vars for Python
ENV PYTHONPATH=/root
ENV PYTHONUNBUFFERED=1

# Expose the Hugging Face port
EXPOSE 7860

# CMD uses the new python orchestrator
CMD ["python", "app/run.py"]