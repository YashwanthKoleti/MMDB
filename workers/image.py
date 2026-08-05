import os
from datetime import datetime, timezone
from src.services import ocr, embedding, text_embedding
from src.database.database import get_files_table, get_images_table
from workers.celery import app

@app.task()
def image_ingestion(file_id, file_path, file_size_bytes, filename=None):    
    vec_384 = None
    ocr_text = None
    
    # vvv imp to make sure to fill ocr_text as None if there is nothing. Because When we ingested images with no text
    # the ingestion code still ran text_embedding("") on the empty string.This generated a valid vector_384 embedding for "" and stores it
    # So, fill it as None
    # There is similar ptroblem in transcipt, but we made sure this error doesnt come by checking if transcipt is empty string or not.
    try:
        extracted = ocr(file_path)
        if extracted and extracted.strip():
            ocr_text = extracted.strip()
            emb = text_embedding(ocr_text)
            if hasattr(emb, "tolist"):
                vec_384 = emb.tolist()
            else:
                vec_384 = emb
    except Exception as e:
        print(f"OCR Error: {e}")
        ocr_text = None
        
    try:
        vector = embedding(file_path, "image")
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
    except Exception as e:
        print(f"Embedding generation failed: {e}")
        vector = None
    
    now = datetime.now(timezone.utc)
    
    files_table = get_files_table()
    files_table.add([{
        "id": file_id,
        "file_name": filename or os.path.basename(file_path),
        "media_type": "image",
        "file_size_bytes": file_size_bytes,
        "upload_date": now,
        "storage_path": file_path
    }])
    
    if vector is not None:
        images_table = get_images_table()
        images_table.add([{
            "id": f"{file_id}_segment_0",
            "file_id": file_id,
            "vector_512": vector,
            "ocr_text": ocr_text,
            "vector_384": vec_384
        }])
    
    return {
        "message": "Image successfully uploaded and processed.",
        "id": file_id,
        "ocr_text": ocr_text
    }

