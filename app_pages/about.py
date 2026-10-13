"""Introduce the app without inventing organization or team details."""
import streamlit as st

st.title("About us", anchor=False)
st.caption("Making HR employment knowledge easier to find and understand.")
st.markdown("The **HR Employment Assistant** helps users explore HR documents through a conversation. It brings the project's PDF library and additional uploaded knowledge into one place, then provides answers with source passages you can check.")

with st.container(border=True):
    st.subheader("Our purpose", divider="orange")
    st.markdown("Help people find relevant information in their HR documents and understand how that information supports an answer.")

st.subheader("What you can do", divider="orange")
st.markdown("""
- Ask questions about the documents in the knowledge library.
- Save PDF sources to the shared document library for future visits.
- Ask follow-up questions in the same conversation.
- Read numbered source passages and their original PDF page references.
""")

st.subheader("Knowledge available", divider="orange")
st.markdown("The initial library covers employment, CPF contributions, workplace safety, tripartite guidelines, and flexible work arrangements. Answers depend on the documents currently available in Chat.")

st.subheader("Using answers thoughtfully", divider="orange")
st.markdown("Check the cited passages and document dates before acting on an answer. The assistant can make mistakes and may lack information needed for your situation. Uploaded PDFs are saved to the shared document library; their text is sent to OpenAI to support search and answers.")
