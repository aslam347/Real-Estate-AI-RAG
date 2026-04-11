from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path

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


def initialize_components():
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",  # latest working model
            temperature=0
        )

    if vector_store is None:
        embedding = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embedding,
            persist_directory=str(VECTORSTORE_DIR)
        )


def process_urls(urls):
    global vector_store

    yield "Initializing Components..."
    initialize_components()

    #  Reset collection safely
    yield "Resetting vector store...✅"
    try:
        vector_store._client.delete_collection(name=COLLECTION_NAME)
    except:
        pass

    # recreate fresh DB
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding,
        persist_directory=str(VECTORSTORE_DIR)
    )

    #  Load content (FIXED)
    yield "Loading data...✅"
    loader = WebBaseLoader(urls)
    data = loader.load()

    #  Split text (IMPROVED)
    yield "Splitting text...✅"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    docs = text_splitter.split_documents(data)

    #  Store in vector DB
    yield "Storing in vector DB...✅"
    ids = [str(uuid4()) for _ in docs]
    vector_store.add_documents(docs, ids=ids)

    yield "Done...✅"


def generate_answer(query):
    if not vector_store:
        raise RuntimeError("Vector DB not initialized")

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    docs = retriever.invoke(query)

    #  Extract context
    context = "\n\n".join([doc.page_content for doc in docs])

    #  Extract sources (IMPORTANT)
    sources = list(set([doc.metadata.get("source", "") for doc in docs]))

    prompt = f"""
    Answer the question using ONLY the context below.

    Context:
    {context}

    Question:
    {query}
    """

    response = llm.invoke(prompt)

    return response.content, sources