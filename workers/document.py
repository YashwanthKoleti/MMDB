import os
from datetime import datetime, timezone
import fitz  
from src.services import embedding, ocr, text_embedding
from src.database.database import get_files_table, get_documents_table, get_images_table
from workers.celery import app

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += (chunk_size - overlap)
    return chunks

# first: take the page as image, run ocr text and then embed the image and ocr text
# seconf: if there are anyinline images, extract them run ocr, and embed them into image and ocr text
# third: take text from the pdf and embed it

@app.task()
def document_ingestion(file_id, file_path, file_size_bytes, filename=None):
    now = datetime.now(timezone.utc)
    ext = os.path.splitext(file_path)[1].lower()
    
    files_table = get_files_table()
    files_table.add([{
        "id": file_id,
        "file_name": filename or os.path.basename(file_path),
        "media_type": "pdf" if ext == ".pdf" else "text",
        "file_size_bytes": file_size_bytes,
        "upload_date": now,
        "storage_path": file_path
    }])
    
    document_table = get_documents_table()
    data = []
    chunk_index = 0
    
    if ext == ".pdf":
        try:
            doc = fitz.open(file_path)
            images_table = get_images_table()
            
            for page_idx, page in enumerate(doc):
                page_number = page_idx + 1
                page_text = page.get_text()
                
                # render and embed full page image
                page_img_path = f"storage/uploads/{file_id}_page_{page_number}.png"
                try:
                    pix = page.get_pixmap(dpi=150)
                    pix.save(page_img_path)
                    
                    page_img_vector = embedding(page_img_path, "image")
                    if hasattr(page_img_vector, "tolist"):
                        page_img_vector = page_img_vector.tolist()
                        
                    #  OCR on the full page image
                    page_ocr_text = ocr(page_img_path)
                    page_ocr_vector = None
                    if page_ocr_text and page_ocr_text.strip():
                        page_ocr_text = page_ocr_text.strip()
                        emb = text_embedding(page_ocr_text)
                        if hasattr(emb, "tolist"):
                            page_ocr_vector = emb.tolist()
                        else:
                            page_ocr_vector = emb
                    else:
                        page_ocr_text = f"Page {page_number} Visual Representation"
                        
                    images_table.add([{
                        "id": f"{file_id}_page_img_{page_number}",
                        "file_id": file_id,
                        "vector_512": page_img_vector,
                        "ocr_text": page_ocr_text,
                        "vector_384": page_ocr_vector
                    }])
                except Exception as e:
                    print(f"Error generating full-page image embedding on page {page_number}: {e}")
                finally:
                    if os.path.exists(page_img_path):
                        os.remove(page_img_path)

                # process inline images
                try:
                    image_list = page.get_images(full=True)
                    for img_idx, img_info in enumerate(image_list):
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        img_path = f"storage/uploads/{file_id}_img_{page_number}_{img_idx}.{image_ext}"
                        with open(img_path, "wb") as f:
                            f.write(image_bytes)
                            
                        # CLIP embedding
                        inline_vector = embedding(img_path, "image")
                        if hasattr(inline_vector, "tolist"):
                            inline_vector = inline_vector.tolist()
                            
                        #  OCR on the inline image
                        inline_ocr_text = ocr(img_path)
                        inline_ocr_vector = None
                        if inline_ocr_text and inline_ocr_text.strip():
                            inline_ocr_text = inline_ocr_text.strip()
                            emb = text_embedding(inline_ocr_text)
                            if hasattr(emb, "tolist"):
                                inline_ocr_vector = emb.tolist()
                        else:
                            inline_ocr_text = None
                            
                        images_table.add([{
                            "id": f"{file_id}_inline_img_{page_number}_{img_idx}",
                            "file_id": file_id,
                            "vector_512": inline_vector,
                            "ocr_text": inline_ocr_text,
                            "vector_384": inline_ocr_vector
                        }])
                        
                        if os.path.exists(img_path):
                            os.remove(img_path)
                except Exception as e:
                    print(f"Error extracting inline images on page {page_number}: {e}")

                # chunk text contents
                if page_text.strip():
                    page_chunks = chunk_text(page_text)
                    for chunk in page_chunks:
                        emb = text_embedding(chunk)
                        if hasattr(emb, "tolist"):
                            emb = emb.tolist()
                            
                        data.append({
                            "id": f"{file_id}_doc_{chunk_index}",
                            "file_id": file_id,
                            "chunk_index": chunk_index,
                            "page_number": page_number,
                            "text_content": chunk,
                            "vector_384": emb
                        })
                        chunk_index += 1
        except Exception as e:
            print(f"Error parsing PDF: {e}")
    else:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                full_text = f.read()
            
            if full_text.strip():
                text_chunks = chunk_text(full_text)
                for chunk in text_chunks:
                    emb = text_embedding(chunk)
                    if hasattr(emb, "tolist"):
                        emb = emb.tolist()
                        
                    data.append({
                        "id": f"{file_id}_doc_{chunk_index}",
                        "file_id": file_id,
                        "chunk_index": chunk_index,
                        "page_number": None,
                        "text_content": chunk,
                        "vector_384": emb
                    })
                    chunk_index += 1
        except Exception as e:
            print(f"Error parsing text file: {e}")
            
    if data:
        document_table.add(data)
        
    return {
        "message": "Document successfully uploaded and processed.",
        "id": file_id,
        "chunks_count": chunk_index
    }
