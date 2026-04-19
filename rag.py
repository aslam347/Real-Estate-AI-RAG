from uuid import uuid4
from dotenv import load_dotenv
import os
import streamlit as st

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# -----------------------------
# CONFIG
# -----------------------------
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "real_estate"

llm = None
vector_store = None


# -----------------------------
# INIT COMPONENTS
# -----------------------------
def initialize_components():
    global llm, vector_store

    groq_key = os.getenv("GROQ_API_KEY")

    if not groq_key:
        try:
            groq_key = st.secrets["GROQ_API_KEY"]
        except:
            raise ValueError("❌ GROQ_API_KEY not found")

    # LLM
    if llm is None:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=groq_key,
            temperature=0
        )

    # EMBEDDINGS
    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    # CHROMA MEMORY MODE (Cloud Safe)
    if vector_store is None:
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embedding
        )


# -----------------------------
# PROCESS URLS
# -----------------------------
def process_urls(urls):
    global vector_store

    yield "Initializing Components...✅"
    initialize_components()

    # Fresh DB every run
    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding
    )

    # Load URLs
    yield "Loading Articles...✅"
    loader = WebBaseLoader(urls)
    data = loader.load()

    # Split Text
    yield "Splitting Content...✅"
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    docs = splitter.split_documents(data)

    # Store
    yield "Creating Knowledge Base...✅"

    ids = [str(uuid4()) for _ in docs]

    vector_store.add_documents(
        documents=docs,
        ids=ids
    )

    yield "Done...✅"


# -----------------------------
# GENERATE ANSWER
# -----------------------------
def generate_answer(query):
    global vector_store, llm

    if vector_store is None:
        raise RuntimeError("Please process URLs first")

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}
    )

    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    sources = list(set([
        doc.metadata.get("source", "")
        for doc in docs if doc.metadata.get("source")
    ]))

    prompt = f"""
You are an expert AI Real Estate Analyst.

Use ONLY the context below.

If answer is missing, say:
Not found in provided articles.

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    return response.content, sources