from lancedb.pydantic import LanceModel, Vector
from typing import Optional
from datetime import datetime

EMBEDDING_DIM = 768

class MultimodalAsset(LanceModel):
    """
    The strict schema for our LanceDB vector database.
    LanceDB will automatically create columns based on these types.
    """
    id: str
    vector: Vector(EMBEDDING_DIM)
    file_path: str
    media_type: str

    # Metadata    
    ocr_text: Optional[str] = None 
    upload_date: datetime
    file_size_bytes: int