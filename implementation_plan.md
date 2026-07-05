# Headless Multimedia Database System

This plan outlines the architecture and implementation steps for a pure backend multimedia database. It focuses entirely on data ingestion, AI-powered embeddings, and cross-modal retrieval, removing any web or frontend dependencies.

## User Review Required

> [!IMPORTANT]
> **Interface Design**: Since you only want to build the database, we need to decide how you'll interact with it. I propose building it as a **Python Library (API)** that you can import and use in scripts (e.g., `db.insert("video.mp4")`, `db.search("sunset", top_k=5)`). Alternatively, we could build a REST API (using FastAPI) without a UI. 
> 
> **Are you comfortable with a Python Library interface, or do you prefer a REST API?**

## Proposed Architecture

The system will act as a unified database engine with three main layers:

1. **Storage Layer**:
   - **Vector Store**: FAISS (for storing and searching dense embeddings).
   - **Metadata Store**: SQLite (for storing file paths, types, dates, and AI-generated text/captions).
   - **Blob Store**: Local filesystem directory for storing the raw media files.
2. **AI Ingestion Pipeline**:
   - **Text**: `sentence-transformers` for text embeddings.
   - **Images**: `CLIP` for image embeddings (aligns with text).
   - **Audio**: `Whisper` (transcription) + `CLAP` (audio-text aligned embeddings).
   - **Video**: Keyframe extraction via `OpenCV` + `CLIP` embeddings.
3. **Query Engine**:
   - Cross-modal routing: Detects if the query is text, image, or audio, generates the corresponding embedding, and queries the vector store.

## Proposed Directory Structure

We will create a clean, modular Python package.

```text
multimedia_db/
├── requirements.txt
├── config.py                 # Configuration (model paths, DB locations)
├── core/
│   ├── engine.py             # Main entry point (the Database class)
│   ├── ingestion/            # Handlers for parsing and chunking media
│   │   ├── image.py
│   │   ├── text.py
│   │   ├── audio.py
│   │   └── video.py
│   ├── models/               # AI Model wrappers (lazy-loaded)
│   │   ├── clip_wrapper.py
│   │   ├── sentence_wrapper.py
│   │   └── clap_wrapper.py
│   ├── retrieval/            # Search logic
│   │   └── cross_modal.py
│   └── storage/              # Database persistence
│       ├── metadata_sqlite.py
│       └── vector_faiss.py
└── tests/
    └── test_basic_flow.py    # Example usage scripts
```

## Implementation Phases

### Phase 1: Core Storage & Basic Ingestion
- Set up the SQLite schema for metadata.
- Set up FAISS indices for vector storage.
- Implement the `Image` and `Text` ingestion pipelines using CLIP and Sentence-Transformers.

### Phase 2: Audio & Video Support
- Integrate Whisper for audio transcription.
- Integrate CLAP for audio embeddings.
- Build the video keyframe extraction pipeline and link it to CLIP.

### Phase 3: The Query Engine
- Build the `Database` class that ties everything together.
- Implement cross-modal search (e.g., searching images using text, searching audio using images).

## Verification Plan

### Automated Tests
- I will create a `demo.py` script that acts as an automated test. It will:
  1. Initialize the database.
  2. Ingest a sample image, text document, and short audio clip.
  3. Perform a text-to-image and text-to-audio search and assert that the correct items are retrieved.

### Manual Verification
- You can provide your own media files, write a short Python script using the library, and verify the retrieval quality.
