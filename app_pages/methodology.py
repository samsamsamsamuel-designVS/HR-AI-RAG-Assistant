"""A beginner guide and reference for the implemented RAG workflow."""
import streamlit as st

st.title("Methodology", anchor=False)
st.caption("This app uses Retrieval-Augmented Generation (RAG). It finds relevant passages in your documents and supplies those passages to an AI model alongside your question.")

st.subheader("How it works", divider="orange")
# Streamlit renders this DOT string with its built-in Graphviz viewer.
st.graphviz_chart(
    r'''digraph RAG {
        graph [rankdir=TB, bgcolor="transparent", pad="0.2", nodesep="0.3", ranksep="0.4"];
        node [shape=box, style="rounded,filled", fillcolor="#303B4B", color="#F0A04B", fontcolor="white", fontname="Arial", fontsize=12, margin="0.18,0.12"];
        edge [color="#F0A04B", fontcolor="#F0A04B", fontname="Arial", fontsize=11];
        subgraph cluster_prepare {
            label="1. Prepare the document library"; fontcolor="#F0A04B"; color="#677587"; style=rounded;
            pdf [label="Preloaded PDFs + saved uploads"];
            text [label="Read PDF pages and split text into passages\nKeep filenames and PDF page numbers"];
            vectors [label="Embed new passages\nRepresent text meaning as numbers"];
            db [label="Chroma search index\nReuse saved passages and embeddings"];
            pdf -> text -> vectors -> db;
        }
        subgraph cluster_answer {
            label="2. Answer your question"; fontcolor="#F0A04B"; color="#677587"; style=rounded;
            question [label="Type a question or click a starter question"];
            query [label="For follow-ups: use recent chat\nto make the search question self-contained"];
            search [label="Embed the search question\nFind up to K relevant passages"];
            answer [label="AI receives your original question,\nselected passages, and recent chat"];
            result [label="Stream an answer with numbered citations"];
            sources [label="View source passages\nCheck cited text and PDF pages"];
            question -> query -> search -> answer -> result -> sources;
        }
        db -> search [label="PDF evidence"];
    }''',
    width=850,
    alt="PDFs are read, split, embedded and saved in Chroma. A question and recent chat guide passage retrieval; the AI streams an answer with citations for the user to check.",
)
st.caption("Preparation reuses cached text and existing embeddings. Each question searches the library; the answer model receives selected passages rather than every PDF.")

steps = (
    ("1. Read and organise the documents",
     "PDFs load from Data and its subfolders, including saved uploads. Text is extracted page by page and split into passages of up to 1,800 characters, with up to 250 characters of overlap to retain context. Each passage keeps its filename and PDF page number. Extracted text is cached on disk for reuse."),
    ("2. Build a searchable library",
     "An embedding is a list of numbers representing the meaning of text. OpenAI's text-embedding-3-small model creates passage embeddings, and Chroma saves them on disk. Only new passages need embedding; removed passages are deleted from the index. Search can match similar ideas even when the wording differs."),
    ("3. Understand and search your question",
     "Starter and typed questions follow the same path. For a follow-up, the answer model first uses up to six recent messages to rewrite it as a standalone search query. Chroma embeds that query and ranks passages using cosine distance, which measures how close their meanings are. The original question is retained for the answer."),
    ("4. Generate an answer from the evidence",
     "The answer model receives your original question, retrieved passages, and up to six recent messages. Its instructions ask it to cite supported statements, explain conflicting evidence, and say when the documents cannot answer. Instructions embedded in PDFs are treated as document content. These safeguards guide the model but cannot guarantee accuracy."),
    ("5. Display and check the result",
     "The answer appears as it is generated. View source passages shows only retrieved passages cited by the answer, preserving labels such as [2], filenames, and PDF page numbers. Answers without citations do not show this panel. PDF page numbers refer to positions in the file and may differ from printed page labels."),
)
for title, explanation in steps:
    with st.container(border=True):
        st.subheader(title)
        st.markdown(explanation)

st.subheader("AI Settings", divider="orange")
st.markdown("""
- **Passages per answer (K):** The maximum passages returned by search, from 1 to 10; the default is 3. More passages provide more evidence to consider but do not guarantee a better answer. If fewer passages exist, search returns what is available.
- **Temperature:** Controls variation in wording, from 0 to 1. The default, 0, favours consistency; it does not guarantee accuracy.
- **Model:** Shows the administrator-configured OpenAI model. This is separate from the embedding model used for search.
""")

st.subheader("Technical reference", divider="orange")
st.markdown("These optional sections map the explanation to the Python code for learning purposes.")

with st.expander("All RAG functions and data structures — rag.py"):
    functions = (
        ("Chunk", "Stores a passage's source filename, PDF page number, and text. A frozen dataclass prevents accidental changes."),
        ("split_text(text, size=1800, overlap=250)", "Normalises whitespace and splits text into overlapping passages; validates the overlap."),
        ("read_pdf(name, content)", "Checks size and page limits, reads PDF bytes, and returns Chunk objects plus extraction warnings. Does not perform OCR."),
        ("read_source_pdf(name, content, directory)", "Loads cached passages for unchanged filenames and PDF bytes, or calls read_pdf and atomically saves a JSON cache."),
        ("SessionIndex", "Holds the Chroma connection, indexed passage IDs, and whether the collection is persistent."),
        ("SessionIndex.close()", "Deletes temporary collections and clears their IDs; leaves persistent app collections intact."),
        ("create_index(api_key, embeddings=None, persist_directory=None)", "Configures embeddings, opens a saved or temporary Chroma collection, and loads existing passage IDs."),
        ("index_chunks(index, chunks)", "Locks shared updates, refreshes saved IDs, and calls _sync_chunks to match the active PDFs."),
        ("_sync_chunks(index, chunks)", "Creates LangChain Documents and stable IDs, removes obsolete passages, and embeds new passages in batches of 64."),
        ("retrieve(index, question, top_k=3)", "Returns up to K (Chunk, similarity) pairs. Converts cosine distance to similarity using 1 minus distance; this is not a confidence percentage."),
        ("cited_passages(answer, evidence)", "Finds valid citations such as [1] or [1, 2], excludes Markdown links, and returns only cited evidence with its original labels."),
        ("create_chat_model(api_key, model, temperature=0.0)", "Creates ChatOpenAI using the Responses API, a 60-second timeout, two retries, a 1,800-token output limit, and store=False."),
        ("search_question(llm, question, history)", "Returns the first question unchanged. With history, rewrites a follow-up using the latest six messages and a 300-token output limit."),
        ("stream_answer(llm, question, history, evidence)", "Labels evidence, builds the answer prompt, and yields text from the LangChain pipeline."),
    )
    for name, description in functions:
        st.markdown(f"**`{name}`** — {description}")
    st.markdown("**Configuration:** `EMBEDDING_MODEL` selects text-embedding-3-small; `PERSIST_DIRECTORY` and `DATABASE_PATH` locate chroma_langchain_db; `SOURCE_LOCK` serialises shared changes; `SYSTEM_PROMPT` supplies evidence and citation instructions.")
    st.code("prompt | llm | StrOutputParser()", language="python")
    st.caption("The | operator connects a prompt, the model, and a plain-text parser. Query rewriting uses invoke(); answers use stream().")

with st.expander("All imports in the RAG engine — rag.py"):
    imports = (
        ("json", "Reads and writes extracted-text caches."),
        ("os", "Atomically replaces cache files with os.replace()."),
        ("re", "Normalises whitespace and identifies numbered citations."),
        ("dataclasses.dataclass", "Defines Chunk and SessionIndex with generated constructors."),
        ("hashlib.sha256", "Creates stable cache keys and passage IDs from content."),
        ("io.BytesIO", "Lets PDF bytes be read as an in-memory file."),
        ("pathlib.Path", "Manages paths to PDFs, the database, and caches."),
        ("tempfile.NamedTemporaryFile", "Writes caches to temporary files before replacement."),
        ("threading.RLock", "Serialises shared library/index changes within the app process."),
        ("uuid.uuid4", "Gives temporary Chroma collections unique names."),
        ("langchain_chroma.Chroma", "Stores embeddings and performs semantic search."),
        ("langchain_core.documents.Document", "Packages passage text with source and page metadata."),
        ("langchain_core.output_parsers.StrOutputParser", "Converts model output into plain text."),
        ("langchain_core.prompts.ChatPromptTemplate", "Builds prompts for query rewriting and answers."),
        ("langchain_core.prompts.MessagesPlaceholder", "Inserts recent conversation into prompts."),
        ("langchain_openai.ChatOpenAI", "Connects LangChain to the OpenAI answer model."),
        ("langchain_openai.OpenAIEmbeddings", "Connects LangChain to the OpenAI embedding model."),
        ("langchain_text_splitters.RecursiveCharacterTextSplitter", "Splits text using spaces, then characters when necessary."),
        ("pypdf.PdfReader", "Reads PDF pages and extracts existing text."),
    )
    for name, description in imports:
        st.markdown(f"**`{name}`** — {description}")

with st.expander("Supporting functions and imports — library.py and app_pages/chat.py"):
    for name, description in (
        ("display_pdf_name", "Hides generated storage hashes in displayed filenames."),
        ("pdf_paths", "Finds PDFs and excludes links outside the library directory."),
        ("save_pdf", "Validates and saves uploads, detects duplicate contents, and caches parsed text."),
        ("delete_pdfs", "Protects preloaded PDFs; deletes selected uploads, indexed passages, and text caches."),
        ("save_uploads", "Handles upload changes, saves files, and queues notices."),
        ("render_library", "Displays the count, document list, and upload-only deletion controls."),
        ("load_local", "Adds Streamlit's in-memory cache around read_source_pdf."),
        ("show_evidence", "Displays cited_passages with filenames and PDF page numbers."),
        ("select_starter_question", "Queues a starter question for the normal chat flow."),
    ):
        st.markdown(f"**`{name}`** — {description}")
    st.markdown("**Library imports:** `json`, `re`, `sha256`, and `Path` handle caches, filenames, hashes, and paths; `Chroma` removes indexed passages; `streamlit as st` draws controls and manages session state. From `rag`: `DATABASE_PATH`, `SOURCE_LOCK`, `read_source_pdf`.")
    st.markdown("**Chat imports:** `os`, `sha256`, and `Path` handle configuration, credential-change detection, and paths; `streamlit as st` draws chat and caches text; `dotenv.dotenv_values` reads administrator settings. OpenAI's `AuthenticationError`, `RateLimitError`, `APIConnectionError`, and `APIStatusError` support readable error messages.")
    st.markdown("**Chat's local imports:** From `library`: `DATA_DIRECTORY`, `display_pdf_name`, `pdf_paths`. From `rag`: `DATABASE_PATH`, `read_source_pdf`, `create_index`, `index_chunks`, `retrieve`, `create_chat_model`, `search_question`, `stream_answer`, `cited_passages`.")

st.subheader("Limits and data handling", divider="orange")
st.markdown("""
- This is a prototype / proof of concept. Answers and citations can be wrong or incomplete. Check the cited text and current official guidance before acting.
- Text-based PDFs are supported up to **20 MB and 1,000 pages per file**. Scans need optical character recognition (OCR) first. Encrypted PDFs that cannot be opened with an empty password are rejected.
- The app searches documents; it does **not browse the web** for updates. Search relevance does not prove correctness or legal applicability.
- Original PDFs, extracted text, and embeddings are saved on disk in a **shared library**. Do not upload sensitive information. Deleting an upload removes its file, passages, and text cache and clears the current chat; other users' existing chat histories are not erased.
- Chat history stays in the current Streamlit session. Document text is sent to OpenAI for embeddings. Questions, recent conversation, and selected passages are sent for query rewriting or answers as applicable. The answer model uses `store=False`.
- The administrator configures the API key and answer model outside the UI. Missing credentials, service limits, or connection problems can prevent chat from becoming available.
""")
