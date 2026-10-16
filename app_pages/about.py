"""Introduce the prototype and help first-time users explore its PDF library."""
import streamlit as st

st.title("About us", anchor=False)
st.caption("Explore HR documents. Ask questions. Check the evidence.")
st.markdown(
    "The **HR AI Employment Assistant** is a prototype / proof of concept built for learning "
    "and demonstration. It helps you explore employment-related PDFs through conversation, "
    "with numbered citations and source passages you can check."
)

with st.container(border=True):
    st.subheader("Our purpose", divider="orange")
    st.markdown(
        "Make HR information easier to find and understand. The assistant uses "
        "**Retrieval-Augmented Generation (RAG)** to search the available documents and "
        "give relevant passages to an AI model to help it answer your question. "
        "Visit **Methodology** for the workflow and technical details."
    )

st.subheader("What's in the knowledge library?", divider="orange")
st.markdown("""
The app starts with three preloaded PDFs:

- **Employment Act** — `Employment-Act.pdf`
- **CPF contributions** — `CPF_Contribution.pdf`
- **Workplace safety and health** — `Workplace_Safety_Health.pdf`

You can add text-based PDFs to test questions about other topics. Answers depend on the documents currently in the library; the app does not browse the web for updates.
""")

st.subheader("How to use the app", divider="orange")
st.markdown("""
1. **Open Chat.** Check **Knowledge sources → Document library** to see the available PDFs.
2. **Ask a question.** Click a starter question or type your own in **Ask a question here...**, then press Enter or the send arrow. Wait for document preparation to finish if the controls are disabled.
3. **Check the evidence.** Read the answer and open **View source passages** to inspect cited text, filenames, and PDF page numbers. Labels such as **[1]** connect the answer to its evidence.
4. **Continue the conversation.** Ask a follow-up or choose **Clear chat** to start again. Clearing chat does not delete documents.
5. **Try another source.** Download a public document as a PDF, then use **Add PDF knowledge** in the sidebar. Wait for it to be saved and prepared before asking about it. Text-based PDFs are supported up to **20 MB and 1,000 pages per file**; scanned images need OCR first.
""")

with st.container(border=True):
    st.subheader("Managing knowledge sources", divider="orange")
    st.markdown("""
**Users can delete uploaded PDFs only. The three preloaded PDFs cannot be deleted.**

To remove an upload, open **Knowledge sources → Document library**, choose it under **Select PDFs to delete**, and click **Delete selected PDFs**. The dropdown lists uploaded documents only. When there are no uploads, the deletion controls are disabled.

Deleting an upload removes it from the shared library, search index, and extracted-text cache, and clears the current chat. Uploaded PDFs are shared with other app users, so removing one also removes it from their available knowledge sources.
""")

st.subheader("Documents to upload and test", divider="orange")
st.markdown("Download the document to your device as a PDF before uploading it. Pasting a website link into Chat does not add it to the knowledge library.")

with st.container(border=True):
    st.subheader("PDPA advisory guidelines")
    st.markdown("Use a public advisory document to explore questions about personal data protection.")
    st.markdown("[Open the PDPC advisory guidelines document](https://www.pdpc.gov.sg/assets/34058be5-ae13-4c40-89e6-1c945d19f65c)")
    st.markdown("""
After uploading, try:

- “What are the main obligations organisations must comply with under the Data Protection Provisions of the PDPA?”
- “How must an organisation obtain valid consent from an individual for the collection, use, or disclosure of personal data under the PDPA?”
""")

with st.container(border=True):
    st.subheader("Retirement and Re-employment Act")
    st.markdown("Use the Act from Singapore Statutes Online to test questions about retirement and re-employment. The link opens a legislation page; obtain a PDF copy before uploading. If needed, use your browser's print function and select **Save as PDF**.")
    st.markdown("[Open the Retirement and Re-employment Act](https://sso.agc.gov.sg/Act/RRA1993)")
    st.markdown("""
After uploading, try:

- “What factors must be considered when varying the terms and conditions in a re-employment contract under the Retirement and Re-employment Act?”
- “What is the minimum duration for a re-employment contract unless otherwise agreed by the parties?”
""")

st.caption("These are suggested test questions, not guaranteed answers. If a detail is not covered by the uploaded document, the assistant should say so. Check the document's version, dates, and cited text.")

st.subheader("Use the assistant thoughtfully", divider="orange")
st.markdown(
    "AI answers may be inaccurate, incomplete, or outdated and do not replace professional HR, "
    "legal, or financial advice. Verify information with current official sources before acting. "
    "Do not enter or upload personal, confidential, or sensitive information. Uploaded PDFs are "
    "saved in a shared library, and document text, questions, and recent conversation are sent "
    "to OpenAI as needed for search and answers."
)
