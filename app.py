"""Run with: python -m streamlit run app.py.

This entry point creates navigation and shared sidebar controls.
The chat interface lives in app_pages/chat.py; rag.py handles PDF and AI logic.
"""
import os
from pathlib import Path

import streamlit as st
from dotenv import dotenv_values

from library import render_library, save_uploads
from utility import account_menu, require_login

# Configure the browser tab and layout before drawing the pages.
st.set_page_config(page_title="HR Employment Assistant ", page_icon=":material/forum:", layout="wide")

# Match page-title underlines to the section dividers, including on login.
# Read the divider palette color so it can be changed in config.toml.
accent_color = st.get_option("theme.orangeColor")
navigation_color = st.get_option("theme.primaryColor")
st.html(f"""
<style>
    h1 {{
        border-bottom: 3px solid {accent_color};
        padding-bottom: 0.5rem;
    }}
    /* aria-current marks the page selected in the sidebar navigation. */
    [data-testid="stSidebarNavLink"][aria-current="page"] {{
        background-color: {navigation_color} !important;
    }}
    [data-testid="stSidebarNavLink"][aria-current="page"] * {{
        color: #FFFFFF !important;
    }}
</style>
""")

# Authenticate before showing any page or accepting uploaded documents.
require_login()

# Display and use the same configured model. Prefer OPENAI_MODEL_NAME in .env;
# also accept the older OPENAI_MODEL setting for existing configurations.
settings = dotenv_values(Path(__file__).resolve().parent / ".env")
st.session_state.model_name = (
    settings.get("OPENAI_MODEL_NAME")
    or os.getenv("OPENAI_MODEL_NAME")
    or settings.get("OPENAI_MODEL")
    or os.getenv("OPENAI_MODEL")
    or "gpt-4.1-mini"
).strip()

# Streamlit reruns this script after interactions. Session state keeps this
# browser session's chat and database connection between those reruns.
# setdefault creates a value only if the key does not already exist.
for key, default in {
    "messages": [],
    "source_index": None,
    "credential_id": None,
}.items():
    st.session_state.setdefault(key, default)

# Navigation shows sidebar links and runs only the selected page.
page = st.navigation(
    [
        st.Page("app_pages/chat.py", title="Chat", icon=":material/forum:", default=True),
        st.Page("app_pages/about.py", title="About us", icon=":material/info:", url_path="about"),
        st.Page("app_pages/methodology.py", title="Methodology", icon=":material/account_tree:", url_path="methodology"),
    ],
    position="sidebar",
)

# Shared widgets live here so their values survive page changes. Uploaded
# files are saved to Data by the callback before Streamlit reruns this script.
with st.sidebar:
    if st.button("Clear chat", icon=":material/delete:", width="stretch", key="clear_chat"):
        st.session_state.messages = []
    st.header(":page_facing_up: Knowledge sources", divider="orange")
    render_library()
    for kind, notice in st.session_state.pop("library_notices", []):
        if kind == "success":
            st.success(notice)
        else:
            st.error(notice)
    upload_key = f"knowledge_uploads_{st.session_state.get('upload_generation', 0)}"
    st.file_uploader(
        "Add PDF knowledge",
        type=["pdf"],
        accept_multiple_files=True,
        max_upload_size=20,
        key=upload_key,
        on_change=save_uploads,
        args=(upload_key,),
    )
    st.caption("Uploaded PDFs are saved to the shared document library for future visits.")

    st.header(":gear: AI Settings  ", divider="orange")
    st.text(f"Model: {st.session_state.model_name}")
    # K is the maximum number of passages retrieved for each question.
    st.slider("Passages per answer (K)", 1, 10, 3, key="passage_count")
    # Temperature controls answer variation: 0 gives more consistent wording.
    st.slider("Temperature", 0.0, 1.0, 0.0, step=0.1, key="temperature")


# Pages own their titles and content. PDF loading and AI requests happen only
# in Chat, so the information pages can open without preparing the knowledge.
page.run()

# Draw account controls last so they appear below the other sidebar sections.
account_menu()
