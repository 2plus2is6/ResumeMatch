from transformers import pipeline
import streamlit as st
from tf_idf import find_matches
from pypdf import PdfReader
import spacy


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


def extractSkills(query_in: str) -> set:

    doc = nlp(query_in.lower())
    skills = set()
    for ent in doc.ents:
        if ent.label_ in {"ORG","PRODUCT","TECH"}:
            skills.add(ent.text.lower())

    for chunk in doc.noun_chunks:
        if chunk.root.dep_ in {"attr","dobj","nsubj"}:
            clean_chunk = " ".join(t.text for t in chunk if not t.is_stop and t.is_alpha)
            if clean_chunk:
                skills.add(clean_chunk.lower()
                           )
    return skills


def get_relevant_sections(text: str, skills: set) -> str:
    doc = nlp(text[:10000])
    section_keywords = ["experience", "skills", "projects", "education"]
    relevant = []

    for sent in doc.sents:
        sent_text = sent.text.lower()

        if any(skill in sent_text for skill in skills) or \
                any(kw in sent_text for kw in section_keywords):
            relevant.append(sent.text)

    return " ".join(relevant[:15])

summarizer_model = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

def summarizer(text: str, query: str) -> str:
    # Extract skills from query
    skills = extractSkills(query)
    relevant_text = get_relevant_sections(text, skills)

    if not relevant_text.strip():
        return "No relevant skills found."

    summary = summarizer_model(
        f"Summarize career highlights focusing on {', '.join(skills)}: {relevant_text}",
        max_length=120,
        min_length=60,
        num_beams=4,
        no_repeat_ngram_size=2,
        truncation=True
    )[0]["summary_text"]

    return ". ".join(list(dict.fromkeys(summary.split(". ")))[:300])



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
        summary = summarizer(cv_texts[name],query_input)
        st.markdown(f"**Summary:** {summary}")
        st.markdown("---")

