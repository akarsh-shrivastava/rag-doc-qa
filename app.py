# app.py
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import streamlit as st
from dotenv import load_dotenv
from rag import chunk_text, load_pdf_with_metadata, build_vector_store, build_vector_store_from_docs, rag_pipeline, rag_pipeline_multi_pdfs

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
    uploaded_files = st.file_uploader("Choose PDFs", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        uploaded_names = sorted([f.name for f in uploaded_files])
        if uploaded_names != st.session_state.get("pdf_names", []):
            with st.spinner("Processing PDFs... this may take a minute."):
                try:
                    pdf_files = [(f.read(), f.name) for f in uploaded_files]
                    st.session_state.vector_store = rag_pipeline_multi_pdfs(pdf_files)
                    st.session_state.pdf_names = uploaded_names
                    st.session_state.chat_history = []  # reset chat on new PDF
                    st.success(f"✅ Ready! {len(uploaded_files)} pages indexed.")

                except Exception as e:
                    st.error(f"❌ Error processing PDF: {e}")
        else:
            st.info(f"📄 {len(uploaded_files)} document(s) loaded.")
            for name in uploaded_names:
                st.caption(f"• {name}")

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
        history_snapshot = list(st.session_state.chat_history)
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.chat_history.append({"role": "user", "content": query})

        # Generate and display answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = rag_pipeline(query, st.session_state.vector_store, history_snapshot)
                    answer = result["answer"]
                    source_docs = result["source_docs"]

                    st.markdown(answer)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer
                    })

                    with st.expander("📚 View Sources"):
                        seen_pages = []
                        for doc in source_docs:
                            page = doc[0].metadata.get("page", "Unknown")
                            source = doc[0].metadata.get("source", "Unknown document")
                            if page not in seen_pages:  # deduplicate same page citations
                                seen_pages.append(page)
                                st.markdown(f"**{source} - Page {page}**")
                                st.caption(doc[0].page_content[:300] + "...")
                                st.divider()

                except Exception as e:
                    st.error(f"❌ Error generating answer: {e}")
