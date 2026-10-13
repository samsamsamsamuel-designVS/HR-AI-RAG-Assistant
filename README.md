# HR knowledge assistant

A Python Streamlit RAG chat app that automatically loads the PDFs in `Data/`, accepts additional PDF uploads, uses LangChain and a Chroma vector store to retrieve relevant passages with OpenAI embeddings, and streams answers with numbered document/page references.

## Run locally

From this project folder in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m streamlit run app.py
```

Before starting the app, edit `.env` in the project folder:

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_MODEL_NAME=gpt-4.1-mini
```

Open the local address printed by Streamlit, log in, and ask a question. The API key stays private. The model is configured in `.env` and displayed in the sidebar as a read-only label. Use a Responses API text model available to your API account. Search uses the separate embedding model `text-embedding-3-small`.

The app automatically reads `.env` beside `app.py` on each rerun. For the model, `OPENAI_MODEL_NAME` in `.env` takes precedence; `OPENAI_MODEL` is also accepted for older configurations. For the API key, existing environment variables take precedence. `.env` is ignored by Git. Use the example above when creating it, and keep the real API key only in `.env` or server environment variables.

## Knowledge and chat

### Login setup

The app requires sign-in before showing any of its pages or accepting uploads.
`utility.py` contains the login guard and the sidebar account/logout menu.
Users enter a username and password, which are checked against `[passwords]` in `.streamlit/secrets.toml`:

```toml
[passwords]
admin = "replace-with-your-password"
```

Each entry defines a username and its password. Your existing secrets file is used directly. To add users, add entries to this section. Restart Streamlit after changing the secrets file. Keep this file private and out of Git; `.gitignore` already excludes it.

OpenAI settings remain in `.env`. Login is remembered in the current session. Logging out clears login and chat history. Saved library PDFs and their persistent embeddings remain available. No external identity provider or OAuth registration is required.

Use the sidebar navigation to switch between **Chat**, **About us**, and **Methodology**. Chat history and AI settings remain available when switching pages. The shared document library is available from the sidebar on every page. Informational pages do not make AI requests.

- PDFs in `Data/` and its subfolders load automatically when you visit Chat. Add or replace a PDF and rerun/refresh Chat to pick it up.
- **Document library** starts collapsed and lists documents with numbers. **Available documents** shows the total number of saved PDFs.
- Upload multiple PDFs in the sidebar. Valid files are saved automatically to `Data/` and appear in the shared document library across sessions and restarts. A hash suffix prevents filename collisions; identical content is saved only once.
- To delete PDFs, expand **Document library**, select files under **Select PDFs to delete**, and click **Delete selected PDFs**. Deletion removes the files, saved Chroma passages, and cached extracted text. It affects every signed-in user and clears the current chat.
- Answers that cite the documents show an expandable source panel containing only the referenced passages, with their original citation numbers and PDF pages. Greetings and answers without valid source citations hide the panel. Citations do not guarantee that every generated claim is correct.
- Follow-up questions use recent conversation context to produce a standalone retrieval query. Clear chat resets the conversation while retaining knowledge.
- Extracted text is cached in memory and on disk using the filename and file contents. Embeddings persist in `./chroma_langchain_db` across sessions and restarts. Only new or changed passages need embedding requests. Removed passages are deleted when the collection is synchronized. Saved uploads and preloaded PDFs use one shared persistent collection.
- Text extraction supports text-based PDFs up to 20 MB and 1,000 pages each. Scans need OCR first. Invalid or password-protected documents show notices without preventing other documents from loading.

## Data flow

PDF bytes → page text → LangChain recursive text splitter → LangChain OpenAI embeddings → Chroma cosine search → LangChain prompt/model/parser pipeline → streamed answer.

RAG means **Retrieval-Augmented Generation**: find relevant document text first, then give that text to an AI model to help answer a question.

1. **Extract:** `pypdf` reads text from each PDF page and preserves the original PDF page number.
2. **Split:** LangChain divides page text into chunks of up to 1,800 characters, with about 250 characters of overlap to preserve context across boundaries.
3. **Embed and save:** An embedding model represents each chunk as a list of numbers. Chroma stores these vectors and the source metadata on disk.
4. **Retrieve:** The question is embedded too. Chroma compares its vector with the saved vectors and returns the closest passages. Follow-up questions are first rewritten using recent conversation context.
5. **Generate:** The answer model receives the retrieved passages, question, and recent chat history. It is instructed to use evidence and cite passages as `[1]`, `[2]`, and so on.
6. **Display:** Streamlit shows the answer as text arrives. The source panel displays only passages with valid citation numbers in the answer; this filtering does not verify whether a claim is supported.

### AI settings

| Setting | Default | Where it is set | What it controls |
| --- | --- | --- | --- |
| Answer model | Value in `.env`; fallback `gpt-4.1-mini` | `OPENAI_MODEL_NAME`, read by `app.py` | Model used for answers and follow-up query rewriting |
| Passages per answer (K) | 3 | Slider in `app.py`, range 1–10 | Maximum number of passages retrieved per question |
| Temperature | 0.0 | Slider in `app.py`, range 0.0–1.0 | Variation in generated wording; lower values are more consistent |
| Embedding model | `text-embedding-3-small` | `EMBEDDING_MODEL` in `rag.py` | Converts passages and search questions into vectors |
| Database directory | `./chroma_langchain_db` | `PERSIST_DIRECTORY` in `rag.py` | Stores embeddings and extracted-text caches |

The chat page passes K to `retrieve()` and temperature to `create_chat_model()`. Temperature support depends on the selected API model. Changing temperature does not change the embedding model or chunk sizes.

### Code guide for beginners

| File | Responsibility |
| --- | --- |
| `app.py` | Entry point: login check, configuration, page navigation, shared sidebar widgets, and session-state defaults |
| `app_pages/chat.py` | Load library text, prepare search, handle questions, stream answers, and display cited sources |
| `rag.py` | Extract and cache PDF text, split chunks, synchronize Chroma, retrieve passages, and build LangChain prompts |
| `library.py` | Save uploads, detect duplicates, list PDFs, and delete files together with their indexed passages and text caches |
| `utility.py` | Check credentials from Streamlit secrets, protect pages, and handle logout |
| `app_pages/about.py` | Describe the app and its purpose |
| `app_pages/methodology.py` | Explain the RAG workflow in the app |
| `tests/` | Verify library, login, retrieval, persistence, and chat behavior without paid AI requests |

Start reading `app.py`, then `app_pages/chat.py`, and finally `rag.py`. Comments explain the flow and Python concepts such as dataclasses, decorators, hashes, and the LangChain `|` pipeline operator.

Streamlit reruns the page script after interactions. `st.session_state` keeps a browser session's login, chat history, widget values, and database connection between reruns. PDFs and embeddings are saved on disk, so they survive logout and app restarts. The app no longer maintains a separate session-only upload index; temporary Chroma collections remain available for tests.

Document text, questions, and recent chat context are sent to OpenAI. API usage requires an API account with billing and model access. Generated responses use `store=False`. Uploaded PDFs are saved in `Data/` as shared app knowledge. Their extracted text and embeddings are cached in `chroma_langchain_db/`, which is ignored by Git. Chat history remains in the current session. Before hosting for multiple users, add access controls and decide which folder documents all users may see.

## Verify

```powershell
.\.venv\Scripts\python -m unittest discover -s tests -v
```

The current suite has 21 tests covering local PDFs, page attribution, unreadable PDF handling, real Chroma retrieval, incremental indexing, removed sources, persistence across processes, temporary collection isolation, LangChain follow-up and streaming pipelines, login, saved uploads and deletion, settings, navigation, and citation display. Tests use deterministic local embeddings and a simulated chat model. They do not make billable requests. Library deletion tests use temporary folders rather than deleting the project's PDFs.

API references: [OpenAI embeddings](https://developers.openai.com/api/docs/guides/embeddings), [OpenAI text generation](https://developers.openai.com/api/docs/guides/text).

Backend references: [LangChain Chroma](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma), [LangChain OpenAI](https://docs.langchain.com/oss/python/integrations/chat/openai).

Persistence is configured by `PERSIST_DIRECTORY = "./chroma_langchain_db"` in `rag.py`, resolved relative to the project folder. The first run extracts and embeds folder PDFs; subsequent runs reuse the cached text and embeddings. Keep this folder to retain the cache.
