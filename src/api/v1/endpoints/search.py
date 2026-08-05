from fastapi import APIRouter
from fastapi.responses import JSONResponse
from src.services import search_text, search_vec

router = APIRouter()

@router.get("/")
def search_api(query: str):
    data = {
        "text": search_text(query),
        "vector": search_vec(query)
    }
    return JSONResponse(content=data)

