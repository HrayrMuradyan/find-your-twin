# Use Python 3.10
FROM python:3.10-slim

# Prevent interactive prompts (Fixes the "debconf" warnings)
ENV DEBIAN_FRONTEND=noninteractive

# 1. Create a new user 'user' with UID 1000
RUN useradd -m -u 1000 user

# 2. Install system dependencies as root
USER root
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. CRITICAL FIX: Create the app directory and assign ownership to 'user'
# Doing this BEFORE WORKDIR ensures 'user' has write permissions.
RUN mkdir -p /home/user/app && chown -R user:user /home/user/app

# 4. Set working directory (Now owned by 'user')
WORKDIR /home/user/app

# 5. Switch to the non-root user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# 6. Copy requirements and install
COPY --chown=user requirements_docker.txt .
RUN pip install --no-cache-dir -r requirements_docker.txt && \
    pip install scrfd --no-deps

# 7. Copy application files
COPY --chown=user pyproject.toml .
COPY --chown=user src ./src
COPY --chown=user models ./models
COPY --chown=user configs ./configs
COPY --chown=user app ./app

# 8. Create writable directories (Now this will succeed)
RUN mkdir -p credentials && \
    mkdir -p .deepface && \
    chmod -R 777 credentials .deepface

# Set Environment Variables
ENV DEEPFACE_HOME="/home/user/app/.deepface"
ENV PYTHONPATH="/home/user/app"
ENV PYTHONUNBUFFERED=1

# Expose the port
EXPOSE 7860

# CMD uses the python orchestrator
CMD ["python", "app/run.py"]