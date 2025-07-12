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
    for token in doc:
        if token.pos_ in {"NOUN","PROPN"} and not token.is_stop:
            skills.add(token.text)
    for chunk in doc.noun_chunks:
        if not any(token.is_stop for token in chunk):
            skills.add(chunk.text.lower())

    #Multi-word skill matching
    from spacy.matcher import Matcher
    matcher = Matcher(nlp.vocab)
    PATTERNS =[
        {"label": "SKILL", "pattern": [{"LOWER": "machine"}, {"LOWER": "learning"}]},
        {"label": "SKILL", "pattern": [{"LOWER": "data"}, {"LOWER": "science"}]},
        {"label": "SKILL", "pattern": [{"LOWER": "cloud"}, {"LOWER": "computing"}]},
        {"label": "SKILL", "pattern": [{"LOWER": "natural"}, {"LOWER": "language"}, {"LOWER": "processing"}]},
    ]
    for pattern in PATTERNS:
        matcher.add(pattern["label"], [pattern["pattern"]])
    matches = matcher(doc)
    for match_id, start, end in matches:
        skills.add(doc[start:end].text.lower())
    return skills


summarizer_model = pipeline("summarization", model="facebook/bart-large-cnn", device="mps")

def summarizer(text: str, query: str) -> str:
    skills = extractSkills(query)
    skills_str = ", ".join(skills) if skills else "relevant skills"

    doc = nlp(text[:10000])
    text = " ".join(sent.text for sent in doc.sents if sent.text.strip())[:4000]  #
    prompt = f"Summarize the following resume in 50–150 words, focusing on skills (e.g., {skills_str}) and experience relevant to: {query}\n\n{text}"

    try:
        summary_cv = summarizer_model(prompt, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
        words = summary_cv.split()
        if len(words) > 150:
            summary_cv = " ".join(words[:150]) + "...."
        elif len(words) < 50:
            summary_cv = summary_cv or text[:200].strip() + "...."
        return summary_cv
    except Exception as e:
        st.warning(f"Error summarizing resume")
        return text[:200].strip() + "...."

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

