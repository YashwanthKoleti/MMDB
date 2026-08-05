# make embedding search
# think about same modal search i.e image to image, audio to audio

import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, ClapModel
from src.database.database import get_images_table, get_audio_table, get_video_table, get_documents_table

# ── Device ──────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Models (lazy-loaded singletons) ────────────────────────────────
_models = {}

def _get_model(key):
    """Lazy-load and cache models so they are only loaded when first needed."""
    if key not in _models:
        if key == "text":
            _models[key] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        elif key == "clip":
            _models[key] = SentenceTransformer("clip-ViT-B-32")
        elif key == "clap_model":
            _models[key] = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(DEVICE)
        elif key == "clap_processor":
            _models[key] = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
        elif key == "qwen":
            _models[key] = SentenceTransformer("Qwen/Qwen3-VL-Embedding-2B")
    return _models[key]


# ── Embedding helpers ──────────────────────────────────────────────

def embed_text_for_text(query: str) -> np.ndarray:
    """Embed a text query for searching transcripts/OCR (384-dim MiniLM)."""
    model = _get_model("text")
    return model.encode(query, convert_to_numpy=True)


def embed_text_for_image(query: str) -> np.ndarray:
    """Embed a text query into CLIP's shared text-image space (512-dim)."""
    model = _get_model("clip")
    return model.encode(query, convert_to_numpy=True)


def embed_text_for_audio(query: str) -> np.ndarray:
    """Embed a text query into CLAP's shared text-audio space (512-dim)."""
    clap_model = _get_model("clap_model")
    clap_processor = _get_model("clap_processor")

    inputs = clap_processor(text=[query], return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        output = clap_model.get_text_features(**inputs)
        if hasattr(output, 'cpu'):
            embedding = output.cpu().numpy()[0]
        else:
            embedding = output.pooler_output.cpu().numpy()[0]

    return embedding


def embed_text_for_video(query: str) -> np.ndarray:
    """Embed a text query into Qwen3-VL's shared text-video space (2048-dim)."""
    model = _get_model("qwen")
    return model.encode(query, convert_to_numpy=True)


# ── Search functions ───────────────────────────────────────────────

# Columns that are not JSON-serializable (numpy arrays, bytes)
_DROP_COLS = [
    "vector_384", "vector_512", "vector_768",
    "vector_1024", "vector_1536", "vector_2048",
    "raw_bytes",
]

def _clean_value(v):
    """Convert a single value to a JSON-safe Python type."""
    import math
    import datetime as dt

    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, bytes):
        return None
    if isinstance(v, (dt.datetime, dt.date)):
        return v.isoformat()
    # pandas Timestamp
    if hasattr(v, 'isoformat'):
        return v.isoformat()
    # pandas NA / NaT
    try:
        import pandas as pd
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v

def _clean_results(df):
    """Drop non-serializable columns and convert all values for JSON output."""
    cols_to_drop = [c for c in _DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    records = df.to_dict(orient="records")
    return [{k: _clean_value(v) for k, v in row.items()} for row in records]


def search_documents(query: str, limit: int = 10):
    """Semantic search over text/PDF documents (MiniLM vector_384)."""
    query_vector = embed_text_for_text(query)
    table = get_documents_table()
    results = (
        table.search(query_vector, vector_column_name="vector_384")
        .metric("cosine")
        .limit(limit)
        .to_pandas()
    )
    return _clean_results(results)


def search_images(query: str, limit: int = 10):
    """CLIP cross-modal search: find images matching text query (vector_512)."""
    query_vector = embed_text_for_image(query)
    table = get_images_table()
    results = (
        table.search(query_vector, vector_column_name="vector_512")
        .metric("cosine")
        .limit(limit)
        .to_pandas()
    )
    return _clean_results(results)


def search_image_ocr(query: str, limit: int = 10):
    """Semantic search over image OCR text (MiniLM vector_384)."""
    query_vector = embed_text_for_text(query)
    table = get_images_table()
    results = (
        table.search(query_vector, vector_column_name="vector_384")
        .metric("cosine")
        .limit(limit)
        .to_pandas()
    )
    return _clean_results(results)


def search_audio(query: str, limit: int = 10):
    """CLAP cross-modal search: find audio clips matching text query (vector_512)."""
    query_vector = embed_text_for_audio(query)
    table = get_audio_table()
    results = (
        table.search(query_vector, vector_column_name="vector_512")
        .metric("cosine")
        .limit(limit)
        .to_pandas()
    )
    return _clean_results(results)


def search_audio_transcript(query: str, limit: int = 10):
    """Semantic search over audio transcripts (MiniLM vector_384)."""
    query_vector = embed_text_for_text(query)
    table = get_audio_table()
    results = (
        table.search(query_vector, vector_column_name="vector_384")
        .metric("cosine")
        .limit(limit)
        .to_pandas()
    )
    return _clean_results(results)


def search_video(query: str, limit: int = 10):
    """Qwen3-VL cross-modal search: find video clips matching text query (vector_2048)."""
    query_vector = embed_text_for_video(query)
    table = get_video_table()
    results = (
        table.search(query_vector, vector_column_name="vector_2048")
        .metric("cosine")
        .limit(limit)
        .to_pandas()
    )

    return _clean_results(results)


def search_video_transcript(query: str, limit: int = 10):
    """Semantic search over video transcripts (MiniLM vector_384)."""
    query_vector = embed_text_for_text(query)
    table = get_video_table()
    results = (
        table.search(query_vector, vector_column_name="vector_384")
        .metric("cosine")
        .limit(limit)
        .to_pandas()
    )
    return _clean_results(results)


def search_all(query: str, limit: int = 10):
    """Unified semantic vector search across all tables and modalities."""
    return {
        "documents": search_documents(query, limit),
        "images": search_images(query, limit),
        "image_ocr": search_image_ocr(query, limit),
        "audio": search_audio(query, limit),
        "audio_transcript": search_audio_transcript(query, limit),
        "video": search_video(query, limit),
        "video_transcript": search_video_transcript(query, limit),
    }



# ── Example usage ──────────────────────────────────────────────────
# from src.database.database import get_table
#
# table = get_table()
# results = search_images("a cat sitting on a couch", table, limit=5)
# print(results[["id", "file_name", "media_type", "_distance"]])
#
# results = search_audio("someone playing guitar", table, limit=5)
# print(results[["id", "file_name", "start_time", "end_time", "_distance"]])
#
# results = search_video("a person cooking in a kitchen", table, limit=5)
# print(results[["id", "file_name", "start_time", "end_time", "_distance"]])