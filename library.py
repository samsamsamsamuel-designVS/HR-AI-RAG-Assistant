"""Manage the shared PDF library on disk and its sidebar controls.

Data stores the original PDFs. Chroma stores searchable embeddings, and
parsed_sources stores extracted text. Deletion removes all three copies.
"""
import json
import re
from hashlib import sha256
from pathlib import Path

from langchain_chroma import Chroma
import streamlit as st

from rag import DATABASE_PATH, SOURCE_LOCK, read_source_pdf

DATA_DIRECTORY = Path(__file__).resolve().parent / "Data"
PRELOADED_PDFS = frozenset({
    "cpf_contribution.pdf",
    "employment-act.pdf",
    "workplace_safety_health.pdf",
})


def display_pdf_name(name):
    """Hide the generated storage hash in user-facing PDF names."""
    return re.sub(r"_[0-9a-f]{16}(?=\.pdf$)", "", name, flags=re.IGNORECASE)


def pdf_paths(directory=None):
    """List PDFs inside the library, excluding links outside its directory."""
    root = Path(directory or DATA_DIRECTORY).resolve()
    if not root.exists():
        return []
    # rglob includes subfolders. Resolve paths to keep links inside the library.
    return sorted(path for path in root.rglob("*")
                  if path.is_file() and path.suffix.lower() == ".pdf"
                  and path.resolve().is_relative_to(root))


def save_pdf(name, content, directory=None, database=None):
    """Return (saved filename, was_created) after validating an upload.

    Comparing file-content hashes detects duplicates even if names differ.
    """
    root = Path(directory or DATA_DIRECTORY).resolve()
    database = Path(database or DATABASE_PATH)
    if not name.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")
    if len(content) > 20 * 1024 * 1024:
        raise ValueError("PDF exceeds the 20 MB limit.")
    digest = sha256(content).hexdigest()
    with SOURCE_LOCK:
        for path in pdf_paths(root):
            if sha256(path.read_bytes()).hexdigest() == digest:
                return path.relative_to(root).as_posix(), False
        # Strip directory parts and unsafe filename characters. A content hash
        # distinguishes different uploads that have the same original filename.
        stem = re.sub(r"[^\w .-]", "_", Path(name.replace("\\", "/")).stem).strip(" .")[:80] or "document"
        filename = f"{stem}_{digest[:16]}.pdf"
        # Parse before saving, rejecting invalid, encrypted, or image-only PDFs.
        # The validated text is also cached for the next page load.
        read_source_pdf(filename, content, database)
        root.mkdir(parents=True, exist_ok=True)
        # xb means write binary bytes, failing if this filename already exists.
        with (root / filename).open("xb") as output:
            output.write(content)
        return filename, True


def delete_pdfs(names, directory=None, database=None):
    """Delete selected PDFs and their saved Chroma passages and parsed caches."""
    root = Path(directory or DATA_DIRECTORY).resolve()
    database = Path(database or DATABASE_PATH)
    # Validate every selected path before deleting anything.
    targets = []
    for name in names:
        target = (root / name).resolve()
        if not target.is_relative_to(root) or target.suffix.lower() != ".pdf" or not target.is_file():
            raise ValueError("Select an existing PDF from the document library.")
        if target.relative_to(root).as_posix().casefold() in PRELOADED_PDFS:
            raise ValueError("Preloaded PDFs cannot be deleted. Select an uploaded PDF.")
        targets.append((target, target.relative_to(root).as_posix()))
    with SOURCE_LOCK:
        # Opening the saved collection for deletion requires no embedding API.
        store = Chroma(collection_name="hr_sources_v1", persist_directory=str(database.resolve())) if (database / "chroma.sqlite3").exists() else None
        for target, source in targets:
            if store is not None:
                ids = store.get(where={"source": source}, include=[])["ids"]
                if ids:
                    store.delete(ids=ids)
            target.unlink()
            # Remove cached text for all saved versions of this source.
            for cache in (database / "parsed_sources").glob("*.json"):
                try:
                    payload = json.loads(cache.read_text(encoding="utf-8"))
                    if any(chunk.get("source") == source for chunk in payload.get("chunks", [])):
                        cache.unlink()
                except (OSError, ValueError, TypeError):
                    continue
    return [source for _, source in targets]


def save_uploads(widget_key):
    """Uploader callback: save files once, then reset the upload widget."""
    notices = []
    for upload in st.session_state.get(widget_key, []) or []:
        try:
            name, created = save_pdf(upload.name, upload.getvalue())
            name = display_pdf_name(name)
            notices.append(("success", f"Saved {name}." if created else f"{name} is already in the library."))
        except Exception as exc:
            notices.append(("error", f"Could not save {upload.name}: {exc}"))
    # The entry point shows these notices after the callback triggers a rerun.
    st.session_state["library_notices"] = notices
    # Use a fresh widget key so deleting a saved PDF cannot accidentally cause
    # its old upload selection to save it again on the next rerun.
    st.session_state["upload_generation"] = st.session_state.get("upload_generation", 0) + 1


def render_library():
    """Show the shared PDF list and a form for deleting selected documents."""
    paths = pdf_paths()
    names = [path.relative_to(DATA_DIRECTORY.resolve()).as_posix() for path in paths]
    deletable_names = [name for name in names if name.casefold() not in PRELOADED_PDFS]
    st.metric("Available documents", len(names))
    with st.expander("Document library", expanded=False):
        for number, name in enumerate(names, start=1):
            st.text(f"{number}. {display_pdf_name(name)}")
        if not names:
            st.caption("Upload a PDF to add it to the library.")
        else:
            st.caption("Only uploaded PDFs can be deleted. Deleting PDFs removes them from the shared library and clears this chat.")
            # A form waits for submit rather than deleting on selection changes.
            with st.form("delete_documents"):
                selected = st.multiselect("Select PDFs to delete", deletable_names, format_func=display_pdf_name, key="delete_selection", disabled=not deletable_names)
                submitted = st.form_submit_button("Delete selected PDFs", icon=":material/delete:", key="delete_pdfs", disabled=not deletable_names)
            if submitted:
                if not selected:
                    st.warning("Select at least one PDF to delete.")
                else:
                    try:
                        deleted = delete_pdfs(selected)
                        # Old answers could cite a document that was just deleted.
                        st.session_state.messages = []
                        st.session_state.library_notices = [("success", f"Deleted {len(deleted)} PDF(s).")]
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not delete the selected PDFs: {exc}")
