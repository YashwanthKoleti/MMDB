from lancedb.pydantic import LanceModel, Vector
from typing import Optional
from datetime import datetime

class FileAsset(LanceModel):
    id: str                       # Unique file ID (UUID)
    file_name: str                # Original name of the uploaded file
    media_type: str               # "image", "audio", "video", "pdf", "text"
    file_size_bytes: int
    upload_date: datetime
    storage_path: str             # Path to the saved file on disk (or raw bytes store reference)

class ImageSegment(LanceModel):
    id: str                       # Unique segment ID
    file_id: str                  # Refers to FileAsset id
    vector_512: Vector(512)       # CLIP image embedding
    ocr_text: Optional[str] = None # Extracted OCR text
    vector_384: Optional[Vector(384)] = None # MiniLM text embedding of OCR text

class AudioSegment(LanceModel):
    id: str                       # Unique segment ID
    file_id: str                  # Refers to FileAsset id
    chunk_index: int
    start_time: float
    end_time: float
    vector_512: Optional[Vector(512)] = None # CLAP audio embedding
    transcript: Optional[str] = None # Whisper speech transcript
    vector_384: Optional[Vector(384)] = None # MiniLM text embedding of transcript

class VideoSegment(LanceModel):
    id: str                       # Unique segment ID
    file_id: str                  # Refers to FileAsset id
    chunk_index: int
    start_time: float
    end_time: float
    vector_2048: Optional[Vector(2048)] = None # Qwen3-VL video chunk embedding
    transcript: Optional[str] = None # Whisper transcript of the audio track for this chunk
    vector_384: Optional[Vector(384)] = None # MiniLM text embedding of transcript


class DocumentSegment(LanceModel):
    id: str                       # Unique segment ID
    file_id: str                  # Refers to FileAsset id
    chunk_index: int
    page_number: Optional[int] = None # Page number (for PDFs)
    text_content: str             # Text content of the chunk
    vector_384: Vector(384)       # MiniLM text embedding of text content