from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def find_matches(cv_texts, userQuery, top_n=5):
    '''top_n=5 will make sure top 5 results are chosen'''

    fileNames = list(cv_texts.keys())
    corpus = list(cv_texts.values())

    corpus.append(userQuery)

    vectorizer = TfidfVectorizer()
    tfidfMatrix = vectorizer.fit_transform(corpus)

    queryVector = tfidfMatrix[-1]
    cvVector = tfidfMatrix[:-1]

    cosineSimilarity = cosine_similarity(queryVector, cvVector)
    similarities = cosineSimilarity.flatten()

    top_indices = np.argsort(similarities)[::-1][:top_n]

    results = [(fileNames[i], round(similarities[i],3)) for i in top_indices]


    return results


