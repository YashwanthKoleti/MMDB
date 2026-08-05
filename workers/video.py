import os
from datetime import datetime, timezone
from src.services.modal.audio import transcribe
from src.services.modal.video import embed_video_chunks
from src.database.database import get_files_table, get_video_table
from workers.celery import app

@app.task()
def video_ingestion(file_id, file_path, file_size_bytes, filename=None):    
    try:
        transcript_chunks = transcribe(file_path)
    except Exception as e:
        print(f"Transcript Error: {e}")
        transcript_chunks = []
        
    try:
        video_chunks = embed_video_chunks(file_path)
    except Exception as e:
        print(f"Embedding generation failed: {e}")
        video_chunks = []
    
    now = datetime.now(timezone.utc)
    
    # parent video file added to the database
    files_table = get_files_table()
    files_table.add([{
        "id": file_id,
        "file_name": filename or os.path.basename(file_path),
        "media_type": "video",
        "file_size_bytes": file_size_bytes,
        "upload_date": now,
        "storage_path": file_path
    }])
    
    video_table = get_video_table()
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
                "vector_2048": None,
                "transcript": chunk["text_content"],
                "vector_384": chunk["vector_384"]
            })
            
    # adding audio chunks to the database
    if video_chunks:
        for chunk in video_chunks:
            chunk_vec = chunk["vector"]
            if hasattr(chunk_vec, "tolist"):
                chunk_vec = chunk_vec.tolist()
                
            data.append({
                "id": f"{file_id}_video_{chunk['id']}",
                "file_id": file_id,
                "chunk_index": int(chunk["id"].split("_")[-1]),
                "start_time": float(chunk["global_chunk_start"]),
                "end_time": float(chunk["global_chunk_end"]),
                "vector_2048": chunk_vec,
                "transcript": None,
                "vector_384": None
            })
            
    if data:
        video_table.add(data)
    
    return {
        "message": "Video successfully uploaded and processed.",
        "id": file_id,
        "transcript_chunks": len(transcript_chunks),
        "video_chunks": len(video_chunks)
    }