# 1. I need to make a pipeline for transcipt.
# 2. I need to make a pipeline for embeddings.

import os
import tempfile

# pyrefly: ignore [missing-import]
from faster_whisper import WhisperModel
import torch
import librosa
from transformers import AutoProcessor, ClapModel
from .embedding import text_embedding

whisper_model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

DEVICE = "cuda" if os.environ.get("USE_CUDA") else "cpu"
compute_type = "float16" if DEVICE == "cuda" else "int8"
clap_processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(DEVICE)

def transcribe(audio_path, chunk_length=30, overlap=5):


    # Whisper expects 16 kHz mono audio
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)

    chunk_samples = int(chunk_length * sr)
    overlap_samples = int(overlap * sr)
    step_samples = chunk_samples - overlap_samples

    documents = []

    chunk_index = 0

    for start in range(0, len(audio), step_samples):
        end = min(start + chunk_samples, len(audio))
        chunk = audio[start:end]

        global_offset = start / sr

        segments, info = whisper_model.transcribe(
            chunk,
            beam_size=5,
        )

        texts = []

        for segment in segments:
            texts.append(segment.text.strip())

        full_text = " ".join(texts).strip()

        if full_text:
            vec = text_embedding(full_text)
            documents.append(
                {
                    "id": f"chunk_{chunk_index}",
                    "global_chunk_start": round(global_offset, 2),
                    "global_chunk_end": round(end / sr, 2),
                    "text_content": full_text,
                    "vector_384": vec
                }
            )

            print(
                f"Chunk {chunk_index}: "
                f"[{global_offset:.1f}s - {end/sr:.1f}s] "
                f"{full_text[:60]}..."
            )

        chunk_index += 1

        if end == len(audio):
            break

    return documents

def embed_audio_chunks(audio_path, chunk_length=10, overlap=3):

    audio, sr = librosa.load(audio_path, sr=48000, mono=True)

    chunk_samples = int(chunk_length * sr)
    overlap_samples = int(overlap * sr)
    step_samples = chunk_samples - overlap_samples

    embeddings = []

    chunk_idx = 0

    for start in range(0, len(audio), step_samples):
        end = min(start + chunk_samples, len(audio))
        chunk = audio[start:end]

        inputs = clap_processor(
            audio=chunk,
            sampling_rate=sr,
            return_tensors="pt"
        )

        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            output = clap_model.get_audio_features(**inputs)
            # Handle both raw tensor and structured output (BaseModelOutputWithPooling)
            if hasattr(output, 'cpu'):
                embedding = output.cpu().numpy()[0]
            else:
                embedding = output.pooler_output.cpu().numpy()[0]

        embeddings.append(
            {
                "id": f"chunk_{chunk_idx}",
                "global_chunk_start": start / sr,
                "global_chunk_end": end / sr,
                "vector": embedding,
            }
        )

        chunk_idx += 1

        if end == len(audio):
            break

    return embeddings

# if __name__ == "__main__":
#     FILE_PATH = "/Users/yashwanth/Desktop/code/Multi Media database/sample.wav" 
    
#     embedding_vectors = embed_audio_chunks(FILE_PATH)
#     print(embedding_vectors)
#     print(transcribe(FILE_PATH))