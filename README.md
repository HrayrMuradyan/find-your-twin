# Find Your Twin

[![Python 3.10.19](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31019/)
[![Database Service Deployment](https://github.com/HrayrMuradyan/find-your-twin/actions/workflows/deploy-database.yml/badge.svg)](https://github.com/HrayrMuradyan/find-your-twin/actions/workflows/deploy-database.yml)
[![Inference Service Deployment](https://github.com/HrayrMuradyan/find-your-twin/actions/workflows/deploy-inference.yml/badge.svg)](https://github.com/HrayrMuradyan/find-your-twin/actions/workflows/deploy-inference.yml)
[![codecov](https://codecov.io/gh/HrayrMuradyan/find-your-twin/graph/badge.svg?token=F6E0R8HT4B)](https://codecov.io/gh/HrayrMuradyan/find-your-twin)

**Author**: Hrayr Muradyan ([LinkedIn](https://www.linkedin.com/in/hrayr-muradyan/))

## About

This project implements a robust Face Similarity Search Engine designed to find "visual twins" or look-alikes from a large dataset of faces.

![Demo](docs/static/vids/demo.gif)

### Highlights

- **Visual Search**: Upload a photo to find the closest matching faces in the database.
- **Image Comparison**: Only faces are compared by automatically finding and croping primary face from any image you upload.
- **Quick Results**: Searches through thousands of images in milliseconds to find the best match.
- **Private Storage**: Securely saves images to Google Drive while keeping search data in a private PostgreSQL database.

### How it Works

The system consists of two main components, each running as a separate microservice:

- **Inference Service (Main)**: Handles the main connection with the website, image processing, and communication with Google Drive.
- **Database Service (Faiss)**: Responsible for the FAISS Vector Database, similarity matching, and communication with PostgreSQL.

**The main similarity matching process is performed as follows:**

Inference Service:

1) The user uploads an image.
2) SCRFD detects the face(s) and keypoints.
3) The primary face is selected based on the area and prediction confidence.
4) Based on the keypoints' eye coordinates, the face is aligned to ensure robust comparison.
5) The DeepFace ArcFace model is used to generate 512-dimensional vector embeddings for the face, normalized for L2 distance.

Database Service:

6) The embeddings are sent to the Database Service, which has FAISS (Facebook AI Similarity Search) loaded.
7) FAISS efficiently retrieves the nearest neighbors based on cosine similarity.
8) The closest matches are sent back to the Inference Service.

Inference Service:

9) Given the matches' metadata, the images are downloaded from Google Drive.
10) The closest matches are sent to the frontend for display.

## Dependencies

The project depends on a modern stack of machine learning and web technologies.

- **Python**: Simpler and powerful programming language 
- **FastAPI**: High-performance web framework for the REST API
- **DeepFace**: A lightweight face recognition and facial attribute analysis framework.
- **SCRFD**: Robust face detection algorithms.
- **FAISS**: A library for efficient similarity search and clustering of dense vectors.
- **Supabase (PostgreSQL)**: Relational database for storing user metadata and persistent vector data.
- **Google Drive API**: Scalable (most importantly free) cloud storage for the image files.

> Note: For specific python package versions, please refer to the [pyproject.toml](pyproject.toml) file.

## Local Development

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

You need to set up the credentials that provide access to your Google Drive and Supabase database.

Create an `.env` file and add the following environment variables to the file:

```bash
# Connection link to Supabase

DB_CONNECTION_STRING= # Ex. postgresql://postgres.{ID}:{PASSWORD}@aws-1-us-east-2.pooler.supabase.com:6543/postgres

# Google Drive client secret and token

GOOGLE_CLIENT_SECRET= # Ex. Secret: {"installed": ...
GOOGLE_TOKEN_JSON= # Ex. Token: {"token": ...

# On which port to run the Inference Service
# Defaults to 7860

INFERENCE_PORT=7860

# On which port to run the Database Port
# Defaults to 8800

DATABASE_PORT=8800
```

### 3. Build with Docker

Build two Docker images using the provided Dockerfiles:

**Inference Service:**

```bash
docker build -f services/inference/Dockerfile -t inference:latest .
```

**Database Service:**

```bash
docker build -f services/database/Dockerfile -t database:latest .
```

### 4. Run the Container

After the containers are built, run both with the following commands:

**Inference Service:**

```bash
docker run -it \
  --env-file .env \
  -p 7860:7860 \
  inference:latest
```

**Database Service:**

```bash
docker run -it \
  --env-file .env \
  -p 8800:8800 \
  database:latest
```

The Inference API will be available at: `http://localhost:7860`.  
The Database API will be available at: `http://localhost:8800`.

### 5. Host the Frontend

To enable real HTTP origin, it's better to run the frontend as a local server:

```bash
python -m http.server 8000 --directory docs
```

The fronend will be available at: `http://localhost:8000`.

## Limitations

Although the tool can quickly and accurately retrieve similar faces from the database, there are a few points to keep in mind:

- **It's All About the Data**: The AI can only find a match within the faces currently in the database. If the database is small or lacks diversity, the "twin" might not look very similar simply because a better match doesn't exist yet.
- **Image Quality Matters**: Lighting, angles, and image resolution play a huge role. A blurry photo or a face covered by sunglasses makes it much harder for the AI to generate an accurate representation.
- **Mathematical Similarity ≠ Human Perception**: The system calculates similarity based on geometric facial features (vectors and cosine similarity). Sometimes, two faces are mathematically similar (e.g., same jawline, eye distance) even if humans wouldn't instinctively call them "look-alikes."
- **No Age or Gender Filters**: The system analyzes facial geometry, not demographics. It does not predict age or gender, so do not be surprised if you are matched with a "twin" of a different age or gender if your features align mathematically. This is something to be corrected in later releases.
- **Limitations of ArcFace and SCRFD**: It's also worth to include the limitations of the two main building blocks of the project.

If your "twin" isn't quite who you hoped for, don't worry! It’s simply the algorithm finding the closest geometric patterns in the data available.


## Contributing

I would love to hear your thoughts, suggestions, and ideas. If you would like to contribute, please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## License

All code in this project is licensed under the [MIT License](LICENSE).


## References

**Tools**:

Van Rossum, G., & Drake, F. L. (2009). Python 3 Reference Manual. Scotts Valley, CA: CreateSpace. https://www.python.org/

Ramírez, S. (2018). FastAPI. https://fastapi.tiangolo.com/

Serengil, S. I., & Ozpinar, A. (2020). LightFace: A Hybrid Deep Face Recognition Framework. In 2020 Innovations in Intelligent Systems and Applications Conference (ASYU). IEEE. https://ieeexplore.ieee.org/document/9259802

Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). ArcFace: Additive Angular Margin Loss for Deep Face Recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). https://arxiv.org/abs/1801.07698

Guo, J., Deng, J., Lian, X., & Shi, H. (2021). Sample and Computation Redistribution for Efficient Face Detection. ArXiv preprint ArXiv:2105.04714. https://arxiv.org/abs/2105.04714

Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. IEEE Transactions on Big Data, 7(3), 535-547. https://arxiv.org/abs/1702.08734

**Datasets**:

Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). Deep Learning Face Attributes in the Wild. In Proceedings of International Conference on Computer Vision (ICCV). http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

Gupta, A. (2023). Human Faces [Data set]. Kaggle. https://www.kaggle.com/datasets/ashwingupta3012/human-faces/data



