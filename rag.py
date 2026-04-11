from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
import os

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Constants
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "real_estate"

llm = None
vector_store = None


# 🔥 Initialize components
def initialize_components():
    global llm, vector_store

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("❌ GROQ_API_KEY not found")

    # ✅ Initialize LLM (FIXED duplicate logic)
    if llm is None:
        llm = ChatGroq(
            model="llama3-70b-8192",
            api_key=groq_key,
            temperature=0
        )

    # ✅ Initialize vector store
    if vector_store is None:
        embedding = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embedding,
            persist_directory=str(VECTORSTORE_DIR)
        )


# 🔥 Process URLs → Load → Split → Store
def process_urls(urls):
    global vector_store

    yield "Initializing Components..."
    initialize_components()

    # ✅ Reset vector DB safely
    yield "Resetting vector store...✅"
    try:
        vector_store._client.delete_collection(name=COLLECTION_NAME)
    except:
        pass

    # ✅ Recreate DB
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding,
        persist_directory=str(VECTORSTORE_DIR)
    )

    # ✅ Load data
    yield "Loading data...✅"
    loader = WebBaseLoader(urls)
    data = loader.load()

    # ✅ Split text
    yield "Splitting text...✅"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    docs = text_splitter.split_documents(data)

    # ✅ Store in DB
    yield "Storing in vector DB...✅"
    ids = [str(uuid4()) for _ in docs]
    vector_store.add_documents(docs, ids=ids)

    yield "Done...✅"


# 🔥 Generate Answer + Sources
def generate_answer(query):
    if not vector_store:
        raise RuntimeError("Vector DB not initialized")

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    docs = retriever.invoke(query)

    # ✅ Context extraction
    context = "\n\n".join([doc.page_content for doc in docs])

    # ✅ Source extraction (clean + unique)
    sources = list(set([
        doc.metadata.get("source", "")
        for doc in docs if doc.metadata.get("source")
    ]))

    # ✅ Strong prompt (better accuracy)
    prompt = f"""
You are a real estate research assistant.

Answer ONLY using the context below.
If answer is not found, say "Not found in context".

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    # ✅ FIX: correct return
    return response.content, sources