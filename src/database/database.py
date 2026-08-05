import os
import lancedb
from .schemas import FileAsset, ImageSegment, AudioSegment, VideoSegment, DocumentSegment

LANCEDB_URI = os.getenv("LANCEDB_URI", "storage/lancedb")

def get_db_connection():
    """
    Returns a LanceDB connection.
    LanceDB handles thread-safe connections under the hood.
    """
    if not LANCEDB_URI.startswith("s3://"):
        os.makedirs(LANCEDB_URI, exist_ok=True)
        
    return lancedb.connect(LANCEDB_URI)

def _get_or_create_table(table_name: str, schema):
    db = get_db_connection()
    if table_name not in db.table_names():
            return db.create_table(table_name, schema=schema)
    else:
            return db.open_table(table_name)

def get_files_table():
    return _get_or_create_table("files", FileAsset)

def get_images_table():
    return _get_or_create_table("image_segments", ImageSegment)

def get_audio_table():
    return _get_or_create_table("audio_segments", AudioSegment)

def get_video_table():
    return _get_or_create_table("video_segments", VideoSegment)

def get_documents_table():
    return _get_or_create_table("document_segments", DocumentSegment)

def init_tables():
    """Pre-create all tables so workers can safely open them without race conditions."""
    print("Initializing LanceDB tables...")
    get_files_table()
    get_images_table()
    get_audio_table()
    get_video_table()
    get_documents_table()
    print("All tables ready.")
