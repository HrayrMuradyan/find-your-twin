# Face Similarity Search / Find Your Twin

[![Python 3.10.19](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31019/)
[![Database Service Deployment](https://github.com/HrayrMuradyan/find-your-twin/actions/workflows/deploy-database.yml/badge.svg)](https://github.com/HrayrMuradyan/find-your-twin/actions/workflows/deploy-database.yml)
[![Inference Service Deployment](https://github.com/HrayrMuradyan/find-your-twin/actions/workflows/deploy-inference.yml/badge.svg)](https://github.com/HrayrMuradyan/find-your-twin/actions/workflows/deploy-inference.yml)
[![codecov](https://codecov.io/gh/HrayrMuradyan/find-your-twin/graph/badge.svg?token=F6E0R8HT4B)](https://codecov.io/gh/HrayrMuradyan/find-your-twin)

Author: Hrayr Muradyan

## About

This project implements a robust Face Similarity Search Engine designed to find "visual twins" or look-alikes from a large dataset of faces.

![Demo](docs/static/vids/demo.gif)

#### How It Works

The system is built as a microservices-style application:
- Upload a face image
- A Face Detection algorithm (defaults to SCRFD) isolates faces from the uploaded image
- A key face is selected among the detected faces
- DeepFace package is used (defaults to ArcFace model) to generate 512-dimensional vector embeddings for the face normalized for L2 distance.
- The embeddings is used to perform vector search in FAISS (Facebook AI Similarity Search) for efficient nearest neighbor retrieval based on cosine similarity.
- Images are stored in Google Drive and metadata/vectors in PostgreSQL.
- The FastAPI is decoupled from the heavy search logic using ZeroMQ for efficient inter-process communication.

## Dependencies

The project depends on a modern stack of machine learning and web technologies.

- **Python**: Simpler and powerful programming language 
- **FastAPI**: High-performance web framework for the REST API
- **DeepFace**: A lightweight face recognition and facial attribute analysis framework.
- **SCRFD**: Robust face detection algorithms.
- **FAISS**: A library for efficient similarity search and clustering of dense vectors.
- **Supabase (PostgreSQL)**: Relational database for storing user metadata and persistent vector data.
- **Google Drive API**: Scalable (most importantly free) cloud storage for the image files.
- **ZeroMQ (pyzmq)**: Asynchronous messaging library used to handle communication between the API and the search service.

> Note: For specific python package versions, please refer to the [requirements_docker.txt](requirements_docker.txt) file.

## Usage

### 1. Clone the Repository

Clone the repository first:

```bash
git clone git@github.com:HrayrMuradyan/find-your-twin.git
```

Move to the root of the repository:

```bash
cd find-your-twin
```

### 2. Setup the Credentials

You need to set up environment variables that provide access to your Google Drive and Supabase database.

Create an `.env` file and `credentials/` folder in the root directory. Add the following environment variables to the `.env` file:

```
DB_CONNECTION_STRING=Access link
GOOGLE_CLIENT_SECRET_PATH=credentials/client_secret.json
GOOGLE_TOKEN_PATH=credentials/token.json
```

Place your `client_secret.json` and `token.json` files in the credentials folder, or provide the content directly to the environment variables.

### 3. Build with Docker

Build the Docker image using the provided Dockerfile:

```bash
docker build -t face-similarity-app .
```

### 4. Run the Container

Run the container, exposing port 7860. Ensure you pass the environment variables and mount the credentials volume if necessary.

```bash
docker run \
  --name face-search \
  -p 7860:7860 \
  --env-file .env \
  -v $(pwd)/credentials:/app/credentials \   # Only if you are providing .json files instead of the content
  face-similarity-app
```

The API will be available at: `http://localhost:7860`.
You can access the interactive API docs at: `http://localhost:7860/docs`.



