# pyrefly: ignore [missing-import]
from fastapi import FastAPI

app = FastAPI()


@app.get("")
async def read_item(item_id):
    return {"item_id": item_id}