from sentence_transformers import SentenceTransformer
from PIL import Image

def image_embedding(image_path:str,model_name:str = "clip-ViT-B-32"):
    model = SentenceTransformer(model_name)
    img = Image.open(image_path)
    image_embeddings = model.encode(img)

    return image_embeddings
    
def text_embedding(text:str, model_name:str  = "sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    text_embeddings = model.encode(text)

    return text_embeddings

def embedding(input1: str, input2: str):
    if input2 == "query":
        return text_embedding(input1)
    else:
        return image_embedding(input1)
