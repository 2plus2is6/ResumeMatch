import streamlit as st
from tf_idf import find_matches
from pypdf import PdfReader
import spacy
nlp = spacy.load('en_core_web_sm')

def preprocess(text: str) -> str:
    doc = nlp(text)
    return " ".join(
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and token.is_alpha
    )



st.title("Resume Matcher")
st.write("Upload multiple CV's and enter keywords to find most relevant candidates.")

fileDataset = st.file_uploader("Upload Files" , accept_multiple_files=True)

keywordInput = st.text_input("Enter description of ideal candidate ",placeholder="Describe the candidate you're looking for (e.g 'Data Scientist with 5 years of Python and NLP experience')" )

if st.button("Search"):
    if not fileDataset or not keywordInput:
        st.write("Please upload at least one file")
    else:
        cv_texts = {}
        for file in fileDataset:
            if not file.name.endswith(".pdf"):
                content = file.read().decode('utf-8').lower()
                cv_texts[file.name] = content
            elif file.name.endswith(".pdf"):
                reader = PdfReader(file)
                content = ""
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        content += text
                cv_texts[file.name] = content.lower()

        query = preprocess(keywordInput)

        sorted_results = find_matches(cv_texts, query)

    # Display results
        st.subheader("CV Match Results")
        for filename, score in sorted_results:
            st.write(f"**{filename}**: - Similarity: {score * 100:.2f}% match")