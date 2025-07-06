import streamlit as st
from tf_idf import find_matches

st.title("Resume Matcher")
st.write("Upload multiple CV's and enter keywords to find most relevant candidates.")

fileDataset = st.file_uploader("Upload Files" , accept_multiple_files=True)

keywordInput = st.text_input("Enter Keywords")
if st.button("Search"):
    if not fileDataset or not keywordInput:
        st.write("Please upload at least one file")
    else:
        keywords = [kw.strip().lower() for kw in keywordInput.split(",")]

        cv_texts = {}
        for file in fileDataset:
            content = file.read().decode('utf-8').lower()
            cv_texts[file.name] = content


        query = keywordInput.strip().lower()
        sorted_results = find_matches(cv_texts, query)

        # Display results
        st.subheader("CV Match Results")
        for filename, score in sorted_results:
            st.write(f"**{filename}**: - Similarity: {score * 100:.2f}% match")