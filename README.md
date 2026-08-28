# Multi-Media Database (MMDB)
#### **MMDB** is a asynchronous multimedia database system designed for multi modal data ingestion, hybrid (keyword + vector) retrieval, and cross-modal semantic search across **Images**, **Audio**, **Video**, and **Documents (PDF & TXT)**.
---
## Key Features
- **Multi-Modal Ingestion**:
  - **Images**: Extracts text using PaddleOCR and generates 512-dim image embeddings using CLIP (`clip-ViT-B-32`) and 384-dim text embeddings using MiniLM (`all-MiniLM-L6-v2`) for extracted text.
  - **Audio**: Transcribes audio using Faster-Whisper, extracts 512-dim audio embeddings using CLAP (`laion/clap-htsat-unfused`), and creates text embeddings for transcripts using MiniLM.
  - **Video**: Transcribes audio tracks using Faster-Whisper, extracts 2048-dim video embeddings using Qwen3-VL (`Qwen/Qwen3-VL-Embedding-2B`), and creates text embeddings for transcripts using MiniLM.
  - **Documents (PDF & TXT)**: Converts PDF pages to images for OCR/CLIP embedding, extracts inline images, chunks document text, and generates 384-dim MiniLM text embeddings.
- **Asynchronous Ingestion Pipeline**:
  - FastAPI endpoints queue file processing jobs to background Celery workers using Redis.
- **Storage**:
  - LanceDB vector database storing vectors and metadata in 5 tables (`files`, `image_segments`, `audio_segments`, `video_segments`, `document_segments`).
- **Hybrid Search (`/v1/search/`)**:
  - **Vector Search**: Cosine similarity search over text, image (CLIP), audio (CLAP), video (Qwen3-VL), and transcript/OCR embeddings.
  - **Keyword Search**: BM25 ranking (`rank_bm25`) with spaCy tokenization (`en_core_web_sm`) and fuzzy query expansion (`rapidfuzz`).
---
## Repository Structure
```text
Multi Media database/
├── docker-compose.yaml          # Redis and LocalStack services
├── scripts/
│   └── download_and_ingest.py   # Batch sample data downloader & ingestion script
├── src/
│   ├── main.py                  # FastAPI application entry point
│   ├── api/                     # REST API routing & endpoints
│   │   └── v1/
│   │       ├── api.py           # API Router (/v1)
│   │       └── endpoints/
│   │           ├── ingestion.py # Media upload endpoints (Image, Audio, Video, Document)
│   │           └── search.py    # Hybrid search API endpoint
│   ├── database/                # LanceDB storage schemas & connections
│   │   ├── database.py          # LanceDB connection & table initializers
│   │   └── schemas.py           # LanceDB Pydantic schemas
│   ├── services/
│   │   ├── modal/               # Feature extractors & ML model wrappers
│   │   │   ├── audio.py         # Whisper transcription & CLAP audio embeddings
│   │   │   ├── embedding.py     # CLIP & MiniLM embedding helpers
│   │   │   ├── ocr.py           # PaddleOCR extraction
│   │   │   └── video.py         # Qwen3-VL video segment embeddings
│   │   └── retrieval/           # Search engines
│   │       ├── keyword_search.py# BM25 + spaCy + RapidFuzz engine
│   │       └── vector_search.py # Multi-modal vector search engine
│   └── ui/
│       └── streamlit_app.py     # Interactive Streamlit dashboard
├── workers/                     # Asynchronous Celery background workers
│   ├── celery.py                # Celery configuration
│   ├── image.py                 # Image processing worker task
│   ├── audio.py                 # Audio processing worker task
│   ├── video.py                 # Video processing worker task
│   └── document.py              # PDF/TXT document worker task
└── storage/                     # Disk storage for file uploads and LanceDB tables
```
---
## Getting Started
### 1. Prerequisites
- **Python**: `3.10` or higher
- **System Dependencies**: `ffmpeg` (for audio/video decoding)
- **Docker**: For running Redis and LocalStack
### 2. Environment Setup
Clone the repository and install dependencies:
```bash
# Clone the repository
git clone https://github.com/YashwanthKoleti/MMDB.git
cd "Multi Media database"
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate
# Install Python requirements
pip install -r requirements.txt
# Download spaCy English language model
python3 -m spacy download en_core_web_sm
```
### 3. Start Infrastructure Services
Spin up Redis (Task Queue) and LocalStack (AWS S3 mock):
```bash
docker-compose up -d
```
---
## Running the Application
To run the complete system, start the Celery worker, FastAPI backend, and Streamlit frontend.
### Step 1: Start Celery Worker
In terminal #1:
```bash
celery -A workers.celery worker --loglevel=info
```
### Step 2: Start FastAPI Application
In terminal #2:
```bash
uvicorn src.main:app --reload --host 127.0.0.1 --port 8000
```
> The API will be available at `http://127.0.0.1:8000`. Interactive documentation is accessible at `http://127.0.0.1:8000/docs`.
### Step 3: Start Streamlit Dashboard
In terminal #3:
```bash
streamlit run src/ui/streamlit_app.py
```
> Access the Web Dashboard at `http://localhost:8501`.
---
## API Endpoints & Usage
### Ingestion Endpoints
- **POST** `/v1/ingestion/image` — Ingest image file (`.png`, `.jpg`, `.jpeg`, `.webp`)
- **POST** `/v1/ingestion/audio` — Ingest audio file (`.wav`, `.mp3`, `.ogg`, `.flac`)
- **POST** `/v1/ingestion/video` — Ingest video file (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`)
- **POST** `/v1/ingestion/document` — Ingest PDF or plain text (`.pdf`, `.txt`)
#### Example cURL:
```bash
curl -X POST "http://127.0.0.1:8000/v1/ingestion/image" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/image.jpg"
```
### Hybrid Search Endpoint
- **GET** `/v1/search/?query={query_string}` — Executes simultaneous BM25 keyword search and multi-modal vector search across all stored assets.
#### Example cURL:
```bash
curl -X GET "http://127.0.0.1:8000/v1/search/?query=landscape"
```
---
## Automated Sample Ingestion Script
You can run the automated script to download sample media (images, audio, video) and ingest them into MMDB:
```bash
python3 scripts/download_and_ingest.py
```
---
## Database Schemas
Data is organized into structured LanceDB tables:
1. **`files`**: Primary asset tracking (ID, original file name, media type, size, upload timestamp, storage path).
2. **`image_segments`**: 512-dim CLIP vectors, extracted OCR text, and 384-dim MiniLM OCR embeddings.
3. **`audio_segments`**: Timestamped chunks with 512-dim CLAP audio vectors, Whisper text transcripts, and 384-dim MiniLM transcript embeddings.
4. **`video_segments`**: Timestamped chunks with 2048-dim Qwen3-VL video vectors, audio track transcripts, and 384-dim MiniLM transcript embeddings.
5. **`document_segments`**: Chunked text content with page indices and 384-dim MiniLM text embeddings.
---
