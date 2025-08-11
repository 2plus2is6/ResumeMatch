import torch
from sentence_transformers import SentenceTransformer

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


model = SentenceTransformer('all-MiniLM-L6-v2')

def find_matches(cv_texts, user_query, top_n=5):
    file_names=list(cv_texts.keys())
    content=list(cv_texts.values())

    if not content:
        return []

    doc_embeddings = model.encode(content)
    query_embeddings = model.encode([user_query])

    similarity_scores = cosine_similarity(query_embeddings, doc_embeddings)[0]

    top_indices = np.argsort(similarity_scores)[::-1][:top_n]

    return [(file_names[i], round(float(similarity_scores[i]), 3)) for i in top_indices]

    return None

