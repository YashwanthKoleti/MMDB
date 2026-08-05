from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.v1.api import api_router

app = FastAPI(title="MMDB")

# Enable CORS for frontend requests (such as Streamlit paste fetch calls)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/v1")

@app.get("/")
async def root():
    return {"message": "Welcome to Multi Media Database API."}