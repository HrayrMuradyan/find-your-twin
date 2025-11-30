# Use Python 3.10
FROM python:3.10-slim

# Set the working directory
WORKDIR /root

# Copy requirements and install them
COPY requirements_docker.txt .

RUN pip install --no-cache-dir -r requirements_docker.txt && \
    pip install scrfd --no-deps

# Install system dependencies (Supervisor + OpenCV libs)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Copy all the files into the container
COPY pyproject.toml /root/pyproject.toml
COPY src /root/src
COPY models /root/models
COPY configs /root/configs
COPY app /root/app

# Copy the docker start script
COPY docker_start.sh /root/docker_start.sh

# Make the start script executable
RUN chmod +x /root/docker_start.sh

# Create a writable directory for temporary files (for deepface)
RUN mkdir -p /root/.deepface && chmod -R 777 /root/.deepface
ENV DEEPFACE_HOME="/root/.deepface"

# Expose the Hugging Face port
EXPOSE 7860

# Start the entrypoint script
CMD ["./docker_start.sh"]