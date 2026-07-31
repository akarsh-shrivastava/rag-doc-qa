import os
import time

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_community.vectorstores import FAISS

load_dotenv()

def chunk_text(text: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.create_documents([text])
    return chunks

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

if __name__ == "__main__":
    sample_pdf = "docs/Laws-of-Cricket-2017-Code-3rd-Edition-2022.pdf"
    text = load_pdf(sample_pdf)

    print("Chunking text...")
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks")

    print("Generating embeddings + building FAISS index...")
    vector_store = build_vector_store(chunks)
    print("Vector store built successfully")

    test_queries = [
        "how many balls in an over",
        "what happens if the ball hits the helmet",
        "when is a batsman given out obstructing the field"
    ]
    for query in test_queries:
        results = vector_store.similarity_search(query, k=3)
        print(f"\nQuery: {query}\n Top result for test query:\n{results[0].page_content}")
