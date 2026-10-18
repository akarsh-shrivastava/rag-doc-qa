# app.py
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import streamlit as st
from dotenv import load_dotenv
from rag import chunk_text, build_vector_store, rag_pipeline

load_dotenv()

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Document Q&A Bot")
st.caption("Upload a PDF and ask questions grounded in its content.")

# ── Session state initialisation ──────────────────────────────
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# ── Sidebar — PDF upload ───────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file is not None:
        # Only reprocess if a new PDF is uploaded
        if uploaded_file.name != st.session_state.pdf_name:
            with st.spinner("Processing PDF... this may take a minute."):
                try:
                    # Read bytes directly — no saving to disk
                    from pypdf import PdfReader
                    import io

                    reader = PdfReader(io.BytesIO(uploaded_file.read()))
                    text = "\n".join(
                        page.extract_text() for page in reader.pages
                        if page.extract_text()
                    )

                    chunks = chunk_text(text)
                    st.session_state.vector_store = build_vector_store(chunks)
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.chat_history = []  # reset chat on new PDF
                    st.success(f"✅ Ready! {len(chunks)} chunks indexed.")

                except Exception as e:
                    st.error(f"❌ Error processing PDF: {e}")
        else:
            st.info(f"📄 Loaded: {st.session_state.pdf_name}")

# ── Main area — Chat interface ─────────────────────────────────
if st.session_state.vector_store is None:
    st.info("👈 Upload a PDF from the sidebar to get started.")
else:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    query = st.chat_input("Ask a question about your document...")

    if query:
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.chat_history.append({"role": "user", "content": query})

        # Generate and display answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = rag_pipeline(query, st.session_state.vector_store)
                    answer = result["answer"]
                    st.markdown(answer)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer
                    })
                except Exception as e:
                    st.error(f"❌ Error generating answer: {e}")
