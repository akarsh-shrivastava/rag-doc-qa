import os
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

def rag_pipeline(query: str, vector_store:FAISS) -> dict:
    context, docs_with_scores = retrieve_chunks(query, vector_store)
    answer = generate_answer(query, context)

    return {
        "question": query,
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