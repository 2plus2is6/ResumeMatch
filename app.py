import os
import nltk
nltk.data.path.insert(0, os.path.join(os.getcwd(), 'nltk_data'))

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

import streamlit as st
import string
from tf_idf import find_matches
from pypdf import PdfReader
import spacy

from operator import itemgetter

# — SpaCy setup for both query preprocessing and stop-words —
nlp = spacy.load("en_core_web_sm")
spacy_stopwords = nlp.Defaults.stop_words

def preprocess(text: str) -> str:
    doc = nlp(text)
    return " ".join(
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and token.is_alpha
    )

#NLTK Summarizer
sents_tokenizer = RegexpTokenizer(r'[^\.!?]+[\.!?]')
words_tokenizer = RegexpTokenizer(r"\b\w+\b")
def summarizer(text: str, top_n=5) -> str:

    sents = sents_tokenizer.tokenize(text)
    words = words_tokenizer.tokenize(text.lower())

    stops = set(stopwords.words("english")) | set(string.punctuation)
    frequencies = {}
    for w in words:
        if w in stops:
            continue
        frequencies[w] = frequencies.get(w, 0) + 1

    sent_scores = {}
    for sent in sents:
        score = sum(frequencies.get(w, 0) for w in words_tokenizer.tokenize(sent.lower()))
        if score > 0:
            sent_scores[sent] = score

    best5 = sorted(sent_scores.items(), key=itemgetter(1), reverse=True)[:top_n]
    summary_cv = " ".join(s for s in sents if s in best5)
    return summary_cv or text[:200].strip() + ".."

# — Streamlit UI —
st.title("Resume Matcher")
st.write("Upload multiple CVs and enter keywords to find the most relevant candidates, with summaries!")

files = st.file_uploader("Upload Files", accept_multiple_files=True)
query_input = st.text_input(
    "Enter description of ideal candidate",
    placeholder="e.g. Data Scientist with 5 years of Python and NLP experience"
)

if st.button("Search"):
    if not files or not query_input:
        st.warning("Please upload at least one file and enter a description.")
        st.stop()

    # Extract raw text
    cv_texts = {}
    for f in files:
        if f.name.lower().endswith(".pdf"):
            reader = PdfReader(f)
            content = "".join(page.extract_text() or "" for page in reader.pages)
        elif f.name.lower().endswith(".txt"):
            content = f.read().decode("utf-8")
        cv_texts[f.name] = content.lower()

    # Preprocess query & rank
    query = preprocess(query_input)
    results = find_matches(cv_texts, query)

    # Display with summaries
    for name, score in results:
        st.subheader(f"{name} — {score*100:.2f}% match")
        summary = summarizer(cv_texts[name], top_n=5)
        st.markdown(f"**Summary:** {summary}")
        st.markdown("---")
