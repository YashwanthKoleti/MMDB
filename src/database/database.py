import os
import lancedb
from .schemas import MultimodalAsset

LANCEDB_URI = os.getenv("LANCEDB_URI", "storage/lancedb")
TABLE_NAME = os.getenv("TABLE_NAME", "multimodal_assets")

def get_db_connection():
    """
    Returns a LanceDB connection.
    LanceDB handles thread-safe connections under the hood.
    """
    if not LANCEDB_URI.startswith("s3://"):
        os.makedirs(LANCEDB_URI, exist_ok=True)
        
    return lancedb.connect(LANCEDB_URI)

def get_table():
    """
    Returns the LanceDB table, creating it if it doesn't exist.
    """
    db = get_db_connection()
    if TABLE_NAME not in db.table_names():
        # Create table with our defined Pydantic schema
        return db.create_table(TABLE_NAME, schema=MultimodalAsset)
    else:
        return db.open_table(TABLE_NAME)
