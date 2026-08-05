import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException

from workers.image import image_ingestion
from workers.audio import audio_ingestion
from workers.video import video_ingestion
from workers.document import document_ingestion

router = APIRouter()

UPLOAD_DIR = "storage/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/image")
async def upload_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1] if file.filename else ".jpg"
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
    
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
        
    file_size_bytes = len(content)

    task = image_ingestion.delay(file_id, file_path, file_size_bytes, file.filename)

    return {
        "message": "Image successfully uploaded and queued.",
        "task_id": task.id,
        "status": task.status
    }
    # vvv imp to make sure to fill ocr_text as None if there is nothing. Because When we ingested images with no text
    # the ingestion code still ran text_embedding("") on the empty string.This generated a valid vector_384 embedding for "" and stores it
    # So, fill it as None
    # There is similar ptroblem in transcipt, but we made sure this error doesnt come by checking if transcipt is empty string or not.


@router.post("/audio")
async def upload_audio(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio.")
    
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1] if file.filename else ".wav"
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
    
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
        
    file_size_bytes = len(content)

    task = audio_ingestion.delay(file_id, file_path, file_size_bytes, file.filename)

    return {
        "message": "Audio successfully uploaded and queued.",
        "task_id": task.id,
        "status": task.status
    }

@router.post("/video")
async def upload_video(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be an video.")
    
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1] if file.filename else ".mp4"
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
    
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
        
    file_size_bytes = len(content)

    task = video_ingestion.delay(file_id, file_path, file_size_bytes, file.filename)

    return {
        "message": "Video successfully uploaded and queued.",
        "task_id": task.id,
        "status": task.status
    }

@router.post("/document")
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".pdf", ".txt"]:
        raise HTTPException(status_code=400, detail="File must be a PDF or text file.")
        
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
    
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
        
    file_size_bytes = len(content)
    
    task = document_ingestion.delay(file_id, file_path, file_size_bytes, filename)
    
    return {
        "message": "Document successfully uploaded and queued.",
        "task_id": task.id,
        "status": task.status
    }