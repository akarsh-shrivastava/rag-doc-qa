import os
import io
import time

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_community.vectorstores import FAISS

load_dotenv()


PROMPT_TEMPLATE = """
You are an expert assistant answering questions about cricket laws.
Use ONLY the context provided below to answer the question.
If the answer is not in the context, say "I cannot find this in the provided document."
Do not make up or infer information not present in the context.

Context:
{context}

Question:
{question}

Answer:
"""

CONDENSE_PROMPT = """Given the chat history and a follow-up question, rewrite the follow-up
as a standalone question that can be understood without the history.
- Keep any specific details from the history that the follow-up depends on.
- If the follow-up is already standalone, return it unchanged.
- Return ONLY the rewritten question. Do not answer it.

Chat history:
{history}

Follow-up question: {question}

Standalone question:
"""


def load_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    pages = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append(text)
        else:
            print(f"Page {page_num + 1}: No text found")
    return "\n".join(pages)

def load_pdf_with_metadata(file_bytes, filename: str) -> list:
    reader = PdfReader(io.BytesIO(file_bytes))
    docs = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            docs.append(Document(
                page_content=text,
                metadata={
                    "page": page_num + 1,
                    "source": filename
                }
            ))
        else:
            print(f"Page {page_num + 1}: No text found")

    if not docs:
        raise ValueError(f"'{filename}' has no extractable text. It may be a scanned (image-based) PDF.")

    return docs

def chunk_text(text: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "\nLAW ",
            "\n\n",
            "\n",
            ".",
            " "
        ]
    )
    chunks = splitter.create_documents([text])
    return chunks

def format_history(chat_history: list, max_messages: int = 6) -> str:
    """Formats only the most recent messages to keep the prompt small."""
    recent = chat_history[-max_messages:]
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in recent)

def condense_question(query: str, chat_history: list) -> str:
    """Rewrites a follow-up into a standalone question. Skips the API call on the first question."""
    if not chat_history:
        return query

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    prompt = CONDENSE_PROMPT.format(
        history=format_history(chat_history),
        question=query
    )
    return llm.invoke(prompt).content.strip()

def build_vector_store(chunks: list) -> FAISS:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    batch_size = 50
    vector_store = None
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"Embedded {min(i + batch_size, len(chunks))}/{len(chunks)} chunks...")
        if vector_store is None:
            vector_store = FAISS.from_documents(batch, embeddings)
        else:
            batch_store = FAISS.from_documents(batch, embeddings)
            vector_store.merge_from(batch_store)
        time.sleep(45)
    
    return vector_store

def build_vector_store_from_docs(docs: list) -> object:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\nLAW ", "\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)  # split_documents preserves metadata
    vector_store = build_vector_store(chunks)
    return vector_store

def rag_pipeline_multi_pdfs(pdf_files: list) -> object:
    vector_store = None

    for file_bytes, filename in pdf_files:
        print(f"Processing {filename}...")
        docs = load_pdf_with_metadata(file_bytes, filename)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\nLAW ", "\n\n", "\n", ".", " "]
        )
        chunks = splitter.split_documents(docs)

        if vector_store is None:
            vector_store = build_vector_store(chunks)
        else:
            new_store = build_vector_store(chunks)
            vector_store.merge_from(new_store)

    return vector_store


def save_vector_store(vector_store: FAISS, path: str = "./faiss_index") -> None:
    vector_store.save_local(path)
    print(f"Vector store saved to {path}")

def load_vector_store(path: str = "./faiss_index") -> FAISS:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    print(f"Vector store loaded from {path}")
    return vector_store


def retrieve_chunks(query: str, vector_store: FAISS, k: int = 3) -> tuple[str,  list[tuple[Document, float]]]:
    docs_with_scores = vector_store.similarity_search_with_score(query, k)
    context_parts = []
    for i, dws in enumerate(docs_with_scores):
        context_parts.append(f"[Chunk {i+1}]    Relevance Score: {dws[1]:.4f}\n{dws[0].page_content}")
    
    context = "\n"
    context += "\n----------------------------------------------\n".join(context_parts)
    return context, docs_with_scores


def generate_answer(query: str, context: str) -> str:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
    formatted_prompt = prompt.format(context=context, question=query)
    response = llm.invoke(formatted_prompt)
    return response.content

def rag_pipeline(query: str, vector_store:FAISS, chat_history: list = None) -> dict:
    chat_history = chat_history or []
    standalone = condense_question(query, chat_history)
    context, docs_with_scores = retrieve_chunks(standalone, vector_store, 6)
    answer = generate_answer(query, context)

    return {
        "question": query,
        "standalone_question": standalone,
        "answer": answer,
        "source_docs": docs_with_scores
    }

if __name__ == "__main__":
    REFERENCE_PDF = "docs/Laws-of-Cricket-2017-Code-3rd-Edition-2022.pdf"
    IS_NEW_PDF = False

    if IS_NEW_PDF or not os.path.exists("./faiss_index"):
        print("Building vector store...")
        text = load_pdf(REFERENCE_PDF)

        print("Chunking text...")
        chunks = chunk_text(text)
        print(f"Created {len(chunks)} chunks")

        print("Generating embeddings + building FAISS index...")
        vector_store = build_vector_store(chunks)
        print("Vector store built successfully")

        save_vector_store(vector_store)
    else:
        print("Loading existing vector store...")
        vector_store = load_vector_store()

    test_queries = [
        # "how many balls in an over",
        # "what happens if the ball hits the helmet",
        # "when is a batsman given out obstructing the field",
        # "how many runs has Sachin Tendulkar scored in ODIs",
        "what happens if a player cheats"
    ]
    for query in test_queries:
        print("\n", "-"*50, sep="")
        result = rag_pipeline(query, vector_store)
        print(f"Q: {result['question']}")
        print(f"A: {result['answer']}")
