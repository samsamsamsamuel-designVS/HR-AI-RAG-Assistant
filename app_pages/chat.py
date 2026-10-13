"""The user interface for the HR knowledge assistant.

Open this page through app.py: python -m streamlit run app.py
This file handles the screen and user interactions. rag.py handles PDF reading,
search, and AI answers. RAG means Retrieval-Augmented Generation: find useful
document passages first, then give them to the AI to help answer the question.
"""
import os
from hashlib import sha256
from pathlib import Path

import streamlit as st
from dotenv import dotenv_values
from openai import AuthenticationError, RateLimitError, APIConnectionError, APIStatusError

from library import DATA_DIRECTORY, pdf_paths
from rag import (
    DATABASE_PATH,
    read_source_pdf,
    create_index,
    index_chunks,
    retrieve,
    create_chat_model,
    search_question,
    stream_answer,
    cited_passages,
)

# Build paths relative to this file so the app works even if launched elsewhere.
ROOT = Path(__file__).resolve().parents[1]
# Read private settings from .env. Server environment variables take priority.
# .strip() removes accidental spaces around the API key.
settings = dotenv_values(ROOT / ".env")
api_key = (os.getenv("OPENAI_API_KEY", settings.get("OPENAI_API_KEY") or "")).strip()
model = st.session_state.model_name
# Streamlit commands create the page elements in the order they run.
st.title("HR AI Employment Assistant")
st.caption("Ask your HR documents. Get clear answers with evidence you can check.")

# This decorator caches the function result based on its input arguments.
@st.cache_data(max_entries=32, show_spinner=False)
def load_local(name, content):
    """Cache folder PDF extraction to avoid reading the same PDF text repeatedly.

    The cache depends on the arguments, so changed file bytes cause fresh parsing.
    Saved uploads and preloaded PDFs are shared app knowledge.
    """
    return read_source_pdf(name, content)


def show_evidence(answer, evidence):
    """Show source passages only when the answer actually cites them."""
    cited = cited_passages(answer, evidence)
    if cited:
        with st.expander("View source passages", icon=":material/library_books:"):
            for i, chunk, _ in cited:
                # Keep the original citation label, even if only [2] was cited.
                st.markdown(f"**[{i}] {chunk.source} · PDF page {chunk.page}**")
                st.text(chunk.text)


# Widgets live in app.py so their values survive navigation to other pages.
top_k = st.session_state.get("passage_count", 3)
temperature = st.session_state.get("temperature", 0.0)

# Flatten each PDF's passages into one list for the shared Chroma collection.
# The in-memory cache and disk cache avoid extracting unchanged PDF text again.
source_chunks = []
paths = pdf_paths(DATA_DIRECTORY)
with st.spinner("Loading knowledge sources…"):
    for path in paths:
        name = path.relative_to(DATA_DIRECTORY).as_posix()
        try:
            chunks, warnings = load_local(name, path.read_bytes())
            source_chunks.extend(chunks)
            for warning in warnings:
                st.warning(warning)
        except Exception as exc:
            # Report a failed PDF while allowing the remaining PDFs to load.
            st.warning(f"{name}: {exc}")

# A fingerprint detects API-key changes without keeping another copy of the
# secret. Reopen the connection with the new key; saved embeddings stay on disk.
credential_id = sha256(api_key.encode()).hexdigest() if api_key else None
if credential_id != st.session_state.credential_id:
    st.session_state.source_index = None
    st.session_state.credential_id = credential_id

llm, ready = None, False
if api_key and source_chunks:
    try:
        llm = create_chat_model(api_key, model, temperature=temperature)
        with st.spinner("Preparing your documents for search…"):
            if st.session_state.source_index is None:
                st.session_state.source_index = create_index(api_key, persist_directory=DATABASE_PATH)
            index_chunks(st.session_state.source_index, source_chunks)
        ready = True
    except AuthenticationError:
        # Show readable messages instead of a Python traceback in the app.
        st.error("The AI service could not authenticate. Please contact the app administrator.")
    except RateLimitError:
        st.error("OpenAI usage limit reached. Check your API billing or try again later.")
    except APIConnectionError:
        st.error("Could not connect to OpenAI. Check your connection and try again.")
    except Exception:
        st.error("Could not prepare the knowledge sources. Check your API settings and retry.")
elif not api_key:
    st.info("Your documents are loaded. The app administrator needs to configure the AI service before chat is available.")

# An empty library must also remove any old passages from the open index.
if not source_chunks and st.session_state.source_index is not None:
    index_chunks(st.session_state.source_index, [])

# A welcome message appears before the first question; it is not sent to the AI.
if not st.session_state.messages:
    with st.chat_message("assistant", avatar=":material/robot_2:"):
        st.markdown("How can I help? Ask about Employment Act Policies, CPF Contributions or Workplace Safety")

# Redraw previous messages on every rerun, including their saved source passages.
for message in st.session_state.messages:
    # Use the same user and assistant icons when replaying saved messages.
    avatar = ":material/face_5:" if message["role"] == "user" else ":material/robot_2:"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        show_evidence(message["content"], message.get("evidence", []))

# Chat stays disabled until search is ready. submit_mode prevents users from
# interrupting an answer by submitting another question while it is streaming.
question = st.chat_input(
    "Ask a question. Example: What are the annual leave entitlements "
    "for employees under the Employment Act?",
    disabled=not ready or not model.strip(),
    submit_mode="disable",
    max_chars=4000,
)
if question:
    # Send only roles and text to the AI, excluding failed turns and UI metadata.
    # Capture history before adding the new question so it is not sent twice.
    history = [
        {"role": message["role"], "content": message["content"]}
        for message in st.session_state.messages
        if not message.get("failed")
    ]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=":material/face_5:"):
        st.markdown(question)
    with st.chat_message("assistant", avatar=":material/robot_2:"):
        try:
            with st.spinner("Finding relevant passages…"):
                # Resolve follow-ups such as "How long does that take?", then
                # retrieve the most relevant chunks from the current library.
                query = search_question(llm, question, history)
                evidence = retrieve(st.session_state.source_index, query, top_k)
            # stream_answer yields small text pieces. write_stream displays them
            # as they arrive and returns the complete text for chat history.
            answer = st.write_stream(stream_answer(llm, question, history, evidence))
            if not answer:
                raise RuntimeError("Empty answer")
            show_evidence(answer, evidence)
            # Save both the answer and its evidence for later page reruns.
            st.session_state.messages.append({"role": "assistant", "content": answer, "evidence": evidence})
        except Exception as exc:
            if isinstance(exc, AuthenticationError):
                error = "The AI service could not authenticate. Please contact the app administrator."
            elif isinstance(exc, RateLimitError):
                error = "OpenAI usage limit reached. Check your API billing or try later."
            elif isinstance(exc, APIStatusError) and exc.status_code == 404:
                error = "The configured AI model is unavailable. Please contact the app administrator."
            else:
                error = "Could not complete the answer. Try again or contact the app administrator."
            st.error(error)
            # Keep the failed turn visible but omit it from future AI context.
            st.session_state.messages[-1]["failed"] = True
            st.session_state.messages.append({"role": "assistant", "content": error, "failed": True})
