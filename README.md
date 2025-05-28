# AR-RAG-Project-Server

## Overview
The AR-RAG-Project-Server is a comprehensive framework designed to facilitate Retrieval-Augmented Generation (RAG) workflows using Milvus, an advanced open-source vector database. This project integrates various tools and frameworks to enable efficient document retrieval, embedding similarity search, and AI-driven applications.

## Features
- **Milvus Integration**: Leverages Milvus for vector search and embedding similarity.
- **RAG Pipelines**: Supports building RAG pipelines with frameworks like Haystack, LangChain, LlamaIndex, DSPy, and FastGPT.
- **OCR Processing**: Extracts text from documents for embedding and retrieval.
- **Cloud Deployment**: Includes guides for deploying Milvus on AWS and other cloud platforms.
- **TLS Encryption**: Provides secure communication with Milvus using TLS.

## Project Structure


### Key Directories
- **documents_RAG/**: Contains source documents for RAG workflows.
- **milvus_docs/**: Documentation for Milvus, including guides for installation, configuration, and integration.
- **output_ocr/**: Stores OCR-processed outputs.
- **volumes/**: Persistent storage for Milvus components like etcd, MinIO, and Milvus.

### Key Files
- **main.py**: Entry point for the server application.
- **milvus-db.ipynb**: Jupyter notebook for Milvus database operations.
- **docker-compose.yml**: Configuration for deploying Milvus and related services using Docker Compose.
- **config.py**: Configuration settings for the project.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/ar-rag-project-server.git
   cd ar-rag-project-server

2. install dependencies
pip install -r requirements.txt

3. copz example.env to .env and update the values as needed

4. start service using Docker Compose
docker-compose up -d

## Usage

1. run the main app
python main.py

2. access mivlus db with milvus-db.ipynb
