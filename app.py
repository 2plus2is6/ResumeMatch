import streamlit as st

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


        match_scores = {}
        for filename, text in cv_texts.items():
            count = 0
            for keyword in keywords:
                if keyword in text:
                    count += 1
            match_scores[filename] = count


        sorted_results = sorted(match_scores.items(), key=lambda x: x[1], reverse=True)

        # Display results
        st.subheader("CV Match Results")
        for filename, score in sorted_results:
            st.write(f"**{filename}**: {score} keyword match{'es' if score != 1 else ''}")