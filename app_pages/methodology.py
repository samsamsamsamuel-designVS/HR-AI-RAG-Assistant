"""Explain the implemented RAG workflow in beginner-friendly language."""
import streamlit as st

st.title("Methodology", anchor=False)
st.caption("How your documents become searchable answers.")
st.markdown("This app uses **Retrieval-Augmented Generation (RAG)**. It finds relevant passages in your documents and supplies those passages to an AI model alongside your question.")

# A list keeps the repeated layout simple: each step has a title and description.
steps = [
    ("1. Load the knowledge sources", "PDFs in the project's Data folder and its subfolders load automatically when you visit Chat. PDFs uploaded through the sidebar are saved to the library for future visits. Duplicate uploads are detected using file contents."),
    ("2. Extract and split the text", "LangChain splits the text extracted from each PDF page into passages of up to 1,800 characters. Neighboring passages overlap by about 250 characters to preserve context. Each passage keeps its document name and original PDF page number."),
    ("3. Represent text as numbers", "An embedding model turns each passage into a list of numbers representing its meaning. These lists make it possible to search for related ideas even when a question uses different words. LangChain sends text to the embedding model, and Chroma saves folder embeddings to disk for reuse across app restarts. Saved uploads join the same persistent collection. Existing passages are reused; removed sources are deleted from the collection."),
    ("4. Find relevant passages", "The question is also converted into an embedding. Chroma searches the stored passages using cosine distance. The app retrieves the number chosen in Passages per answer: 1 to 10, with a default of 3. If fewer passages exist, it returns the available ones."),
    ("5. Answer from the evidence", "A LangChain prompt and model pipeline receives the question, retrieved passages, and recent conversation. Its instructions ask it to answer from the supplied evidence, use numbered citations, and acknowledge missing or conflicting information. Follow-up questions are first rewritten as standalone search queries using recent chat context."),
    ("6. Show the answer and sources", "The answer appears as it is generated. When it cites documents, View source passages shows only the referenced passages, with their citation numbers, document names, and PDF page numbers. Greetings and answers without source citations do not show this panel."),
]
for title, description in steps:
    with st.container(border=True):
        st.subheader(title, divider="orange")
        st.markdown(description)

st.subheader("Limits and data handling", divider="orange")
st.markdown("""
- Text-based PDFs are supported up to 20 MB and 1,000 pages per file. Scanned pages need optical character recognition (OCR) first.
- Search ranks passages by relevance. Similarity scores and citations do not guarantee that an answer is correct or complete.
- The app does not check the web for updates. An answer reflects the available source documents.
- Uploaded PDFs are saved to the shared library. All library documents have extracted text and embeddings cached on disk for faster startup. Chat history stays in the current session. Selected PDFs can be deleted from Document library; deletion also removes their indexed passages.
- Document text, questions, and recent conversation are sent to OpenAI. The API key and answer model are configured by the administrator outside the UI.
""")
