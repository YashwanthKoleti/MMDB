from fastapi import APIRouter
from .endpoints import ingestion, search

api_router = APIRouter()

api_router.include_router(ingestion.router, prefix="/ingestion", tags=["ingestion"])
api_router.include_router(search.router, prefix="/search", tags=["search"])
