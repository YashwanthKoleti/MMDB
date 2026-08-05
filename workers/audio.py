import os
from datetime import datetime, timezone
from src.services.modal.audio import transcribe, embed_audio_chunks
from src.database.database import get_files_table, get_audio_table
from workers.celery import app

@app.task()
def audio_ingestion(file_id, file_path, file_size_bytes, filename=None):    
    try:
        transcript_chunks = transcribe(file_path)
    except Exception as e:
        print(f"Transcript Error: {e}")
        transcript_chunks = []
        
    try:
        audio_chunks = embed_audio_chunks(file_path)
    except Exception as e:
        print(f"Embedding generation failed: {e}")
        audio_chunks = []
    
    now = datetime.now(timezone.utc)
    
    # 1. Add global file registry entry
    files_table = get_files_table()
    files_table.add([{
        "id": file_id,
        "file_name": filename or os.path.basename(file_path),
        "media_type": "audio",
        "file_size_bytes": file_size_bytes,
        "upload_date": now,
        "storage_path": file_path
    }])
    
    audio_table = get_audio_table()
    data = []
    
    #adding transcipt chunks into databse
    if transcript_chunks:
        for chunk in transcript_chunks:
            data.append({
                "id": f"{file_id}_transcript_{chunk['id']}",
                "file_id": file_id,
                "chunk_index": int(chunk["id"].split("_")[-1]),
                "start_time": float(chunk["global_chunk_start"]),
                "end_time": float(chunk["global_chunk_end"]),
                "vector_512": None,
                "transcript": chunk["text_content"],
                "vector_384": chunk["vector_384"]
            })
            
    # adding audio chunks to the database
    if audio_chunks:
        for chunk in audio_chunks:
            chunk_vec = chunk["vector"]
            if hasattr(chunk_vec, "tolist"):
                chunk_vec = chunk_vec.tolist()
                
            data.append({
                "id": f"{file_id}_audio_{chunk['id']}",
                "file_id": file_id,
                "chunk_index": int(chunk["id"].split("_")[-1]),
                "start_time": float(chunk["global_chunk_start"]),
                "end_time": float(chunk["global_chunk_end"]),
                "vector_512": chunk_vec,
                "transcript": None,
                "vector_384": None
            })
            
    if data:
        audio_table.add(data)
    
    return {
        "message": "Audio successfully uploaded and processed.",
        "id": file_id,
        "transcript_chunks": len(transcript_chunks),
        "audio_chunks": len(audio_chunks)
    }