# app.py
import os
import io
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import streamlit as st
from dotenv import load_dotenv
from rag import get_sample_vector_store, rag_pipeline, rag_pipeline_multi_pdfs

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_docs")
MAX_UPLOAD_MB = 15

SAMPLE_QUESTIONS = [
    "What does Student membership cost per year?",
    "I'm a student member. How much for 2 hours on the nets on a Saturday?",
    "Can my 12-year-old use the nets alone?",
]


# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Document Q&A Bot")
st.caption("Upload a PDF and ask questions grounded in its content.")

# ── Session state initialisation ──────────────────────────────
for key, default in {
    "vector_store": None,
    "chat_history": [],
    "pdf_names": [],
    "using_sample": False,
    "pdf_bytes": {}
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

def error_handler(e: Exception) -> str:
    msg = str(e).lower()
    if isinstance(e, ValueError):  # our own errors, e.g. scanned PDF
        return f"📄 {e}"
    if "resource_exhausted" in msg or "429" in msg or "quota" in msg:
        return "⏳ The Gemini API rate limit was hit. Please wait a minute and try again."
    if "api key" in msg or "api_key" in msg or "permission_denied" in msg:
        return "🔑 There's a problem with the API key configuration."
    return "⚠️ Something went wrong. Please try again."

def build_sources(docs) -> list:
    seen, out = set(), []
    for d in docs:
        key = (d[0].metadata.get("source"), d[0].metadata.get("page"))
        if key not in seen:
            seen.add(key)
            out.append({
                "source": d[0].metadata.get("source", "Unknown document"),
                "page": d[0].metadata.get("page", "?"),
                "snippet": d[0].page_content[:300],
            })
    return out

def show_sources(sources):
    if not sources:
        return
    with st.expander(f"📚 View Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**{s['source']} — Page {s['page']}**")
            st.caption(s["snippet"] + "...")
            st.divider()

def index_documents(pdf_files: list, names: list, is_sample: bool = False):
    with st.spinner(f"Indexing {len(pdf_files)} document(s)... this can take a minute."):
        try:
            st.session_state.vector_store = rag_pipeline_multi_pdfs(pdf_files)
            st.session_state.pdf_bytes = {name: data for data, name in pdf_files}  # NEW
            st.session_state.pdf_names = names
            st.session_state.chat_history = []
            st.session_state.using_sample = is_sample

        except Exception as e:
            print(f"[index error] {e!r}")  # full detail goes to the server log
            st.error(error_handler(e))

@st.cache_resource(show_spinner="Indexing sample documents (first visit only)...")
def get_sample_store():
    files = []
    for name in sorted(os.listdir(SAMPLE_DIR)):
        if name.lower().endswith(".pdf"):
            with open(os.path.join(SAMPLE_DIR, name), "rb") as f:
                files.append((f.read(), name))
    return get_sample_vector_store(files), files

def render_document_viewer():
    names = st.session_state.pdf_names
    open_by_default = not st.session_state.chat_history

    with st.expander("📖 View documents", expanded=open_by_default):
        labels = [n.replace(".pdf", "").replace("_", " ").title() for n in names]
        for tab, name in zip(st.tabs(labels), names):
            with tab:
                data = st.session_state.pdf_bytes.get(name)
                if data is None:
                    st.caption("Preview not available.")
                    continue
                try:
                    st.pdf(io.BytesIO(data), height=450)
                except Exception:
                    st.download_button(
                        "⬇️ Download PDF", data, file_name=name,
                        mime="application/pdf", key=f"dl_{name}"
                    )

# ── Sidebar — PDF upload ───────────────────────────────────────
with st.sidebar:
    st.header("📂 Documents")
    uploaded_files = st.file_uploader(
        "Choose one or more PDFs", type="pdf", accept_multiple_files=True
    )

    if uploaded_files:
        total_mb = sum(f.size for f in uploaded_files) / (1024 * 1024)
        if total_mb > MAX_UPLOAD_MB:
            st.error(f"Total upload is {total_mb:.1f} MB. Limit is {MAX_UPLOAD_MB} MB.")
        else:
            names = sorted(f.name for f in uploaded_files)
            if names != st.session_state.pdf_names:
                index_documents([(f.getvalue(), f.name) for f in uploaded_files], names)
    else:
        st.caption("No PDF handy?")
        if st.button("✨ Try with sample documents"):
            try:
                store, files = get_sample_store()
                st.session_state.vector_store = store
                st.session_state.pdf_names = [n for _, n in files]
                st.session_state.pdf_bytes = {n: b for b, n in files}
                st.session_state.chat_history = []
                st.session_state.using_sample = True
            except Exception as e:
                print(f"[sample error] {e!r}")
                st.error(error_handler(e))

    if st.session_state.vector_store is not None:
        st.divider()
        st.subheader("Loaded")
        for n in st.session_state.pdf_names:
            st.caption(f"📄 {n}")
        st.caption(f"{st.session_state.vector_store.index.ntotal} chunks indexed")
        if st.button("🗑️ Clear chat"):
            st.session_state.chat_history = []
            st.rerun()


# ── Main area — Chat interface ─────────────────────────────────

if st.session_state.vector_store is None:
    st.info("👈 Upload one or more PDFs, or try the sample documents, to get started.")
    st.stop()

render_document_viewer()

for m in st.session_state.chat_history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        show_sources(m.get("sources"))

pending = None
if st.session_state.using_sample and not st.session_state.chat_history:
    st.caption("Try one of these:")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, key=f"example_{q}"):
            pending = q

query = st.chat_input("Ask a question about your documents...") or pending

if query:
    history_snapshot = list(st.session_state.chat_history)  # before appending the new question

    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.chat_history.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the documents..."):
                result = rag_pipeline(query, st.session_state.vector_store, chat_history=history_snapshot)
        except Exception as e:
            print(f"[answer error] {e!r}")
            st.error(error_handler(e))
            st.session_state.chat_history.pop()  # drop the unanswered question
        else:
            answer = result["answer"]
            # Don't show sources under a refusal: they'd imply the answer came from them
            sources = [] if answer.startswith("I cannot find") else build_sources(result["source_docs"])

            st.markdown(answer)
            if result["standalone_question"] != query:
                st.caption(f"🔎 Searched as: {result['standalone_question']}")
            show_sources(sources)

            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )