import spacy
from rank_bm25 import BM25Okapi
from rapidfuzz import process
from src.database.database import get_images_table, get_audio_table, get_video_table, get_documents_table

nlp = spacy.load("en_core_web_sm")

# Example
# documents = [
#     "the quick brown fox jumps over the lazy dog near the river bank",
#     "machine learning and deep learning are subfields of artificial intelligence",
#     "the stock market crashed due to rising inflation and interest rates",
#     "climate change is causing rising sea levels and extreme weather events",
#     "python is a popular programming language used in data science and web development",
# ]

# ocr_texts = [
#     "Invoice #1042 Total Amount $250.00 Date 2025-01-15",
#     "WARNING: Do not operate heavy machinery while taking this medication",
#     "Chapter 3: Introduction to Neural Networks and Backpropagation",
#     "Annual Report 2024 Revenue Growth 15% Net Profit $2.3M",
#     "def train_model(data): model.fit(data) return model",
# ]

def tokenize(text):
    doc = nlp(text)
    tokens = [t.text for t in doc]
    return tokens
    
def get_vocab(tokenized_docs):
    vocabulary = set()

    for tokens in tokenized_docs:
        vocabulary.update(tokens)

    vocabulary = list(vocabulary)
    return vocabulary

def build_index(tokenized_docs):
    bm25 = BM25Okapi(tokenized_docs)
    return bm25


def search(query, bm25, vocabulary):
    tokenized_query = tokenize(query)
    expanded = []

    for word in tokenized_query:

        matches = process.extract(
            word,
            vocabulary,
            limit=3,
            score_cutoff=70
        )

        expanded.extend([match for match, score, idx in matches])

    expanded_query = list(set(tokenized_query + expanded))
    scores = bm25.get_scores(expanded_query)

    best = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True
    )

    return best

# ig this is a single function that query and list of texts and run bm25, instead of writing the same code for ocr, then for audio transcipt, then for documnets - 
# - this helper function will be used to reduce repeatability
def _bm25_search_on_list(query, text_list):
    non_empty = [t for t in text_list if t and str(t).strip()]
    if not non_empty:
        return []
    tokenized_docs = [tokenize(doc) for doc in non_empty]
    vocabulary = get_vocab(tokenized_docs)
    bm25 = build_index(tokenized_docs)
    
    # run the search
    ranked_indices = search(query, bm25, vocabulary)
    
    results = []
    for rank_idx, score in ranked_indices:
        if score > 0:  # Only return matching items
            results.append({
                "text": non_empty[rank_idx],
                "score": float(score)
            })
    return results

def search_documents_keyword(query):
    df = get_documents_table().to_pandas()
    return _bm25_search_on_list(query, df['text_content'].tolist())

def search_audio_transcript_keyword(query):
    df = get_audio_table().to_pandas()
    return _bm25_search_on_list(query, df['transcript'].tolist())

def search_video_transcript_keyword(query):
    df = get_video_table().to_pandas()
    return _bm25_search_on_list(query, df['transcript'].tolist())

def search_ocr_keyword(query):
    df = get_images_table().to_pandas()
    return _bm25_search_on_list(query, df['ocr_text'].tolist())

def search_text(query):
    return {
        "documents": search_documents_keyword(query),
        "transcript_audio": search_audio_transcript_keyword(query),
        "transcript_video": search_video_transcript_keyword(query),
        "ocr": search_ocr_keyword(query)
    }