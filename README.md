# 🏡 AI Real Estate Intelligence Engine  
### Gen AI + RAG Project using Llama 3.3, GROQ, LangChain & Streamlit

An intelligent **AI-powered Real Estate Research Assistant** that analyzes real estate news articles and market reports from live URLs, then answers user questions using **Retrieval-Augmented Generation (RAG)**.

This project helps users quickly understand housing market trends, luxury property pricing, investment insights, and real estate news through conversational AI.

---

# 🌐 Live Demo

🚀 Try the app here:

👉 https://real-estate-ai-rag-mohamed-aslam.streamlit.app/

---

# 🎥 Project Demo Video

Watch full explanation and walkthrough:

👉 https://www.loom.com/share/45c328b138fe4e6e841d16dc4baee01d

---

# 💻 GitHub Repository

👉 https://github.com/aslam347/Real-Estate-AI-RAG

---

# 📌 Project Overview

Users can paste real estate article URLs from websites like:

- CNBC  
- Forbes  
- Zillow  
- Realtor.com  
- HousingWire  
- MarketWatch  
- Any public real estate article  

The AI system:

✅ Extracts article content  
✅ Converts text into embeddings  
✅ Stores chunks in vector database  
✅ Understands user questions  
✅ Retrieves relevant context  
✅ Generates intelligent answers with sources  

---

# 🚀 Key Features

✅ Live URL article processing  
✅ AI answers from multiple real estate articles  
✅ RAG architecture with source grounding  
✅ Real estate market intelligence engine  
✅ Investment research assistant  
✅ Luxury housing trend analysis  
✅ Source links for transparency  
✅ Fast responses powered by GROQ  
✅ Premium Streamlit UI  
✅ Cloud deployed application  
✅ Docker containerization support  

---

# 🧠 Example Questions

- What is happening in the U.S. luxury housing market?  
- Which cities have the most million-dollar listings?  
- Summarize this article in simple words  
- Is the market good for investors now?  
- Compare two housing market articles  
- What trends are mentioned in the report?  
- Which region shows strongest growth?  

---

# 🛠️ Tech Stack

- Python  
- Streamlit  
- LangChain  
- GROQ API  
- Llama 3.3  
- ChromaDB  
- HuggingFace Embeddings  
- BeautifulSoup  
- WebBaseLoader  
- Docker  

---

# 🏗️ Architecture Flow

```text
User URLs
   ↓
Web Scraping / Loader
   ↓
Text Chunking
   ↓
Embeddings Generation
   ↓
Vector Database
   ↓
User Question
   ↓
Retriever
   ↓
Llama 3.3 via GROQ
   ↓
Final AI Answer + Sources
```

---

# 📁 Folder Structure

```bash
Real-Estate-AI-RAG/
│
├── main.py
├── rag.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── README.md
│
├── resources/
│   └── screenshots/
│
└── venv/
```

---

# ⚙️ Installation & Run

```bash
git clone https://github.com/aslam347/Real-Estate-AI-RAG.git
cd Real-Estate-AI-RAG
pip install -r requirements.txt
streamlit run main.py
```

---

# 🐳 Docker Containerization

This project is containerized using Docker for easy deployment and sharing.

## Build Docker Image

```bash
docker build -t real-estate-ai-rag .
```

## Run Docker Container

```bash
docker run --env-file .env -p 8501:8501 real-estate-ai-rag
```

## Open in Browser

```text
http://localhost:8501
```

---

# 🐳 Docker Hub

Pull and run directly from Docker Hub:

```bash
docker pull mohamedaslam2001/real-estate-ai-rag
docker run --env-file .env -p 8501:8501 mohamedaslam2001/real-estate-ai-rag
```

---

# 🔐 Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

---

# 💡 Real Business Use Cases

✅ Real estate market research  
✅ Property investment intelligence  
✅ Housing news summarizer  
✅ Realtor AI assistant  
✅ Luxury market trend analysis  
✅ Real estate consulting support  
✅ Competitive market monitoring  

---

# 🔥 Challenges Solved

During development, this project involved solving real-world engineering issues such as:

- Cloud deployment on Streamlit  
- ChromaDB compatibility fixes  
- Git conflict resolution  
- LLM model migration after deprecation  
- Live article processing pipeline  
- Production-ready RAG workflow  
- Docker containerization  
- Environment variable management  

---

# 📚 Key Learnings

- Retrieval-Augmented Generation (RAG)  
- LangChain pipeline development  
- Embedding generation and retrieval  
- ChromaDB vector database usage  
- Prompt engineering  
- GenAI application deployment  
- Dockerizing AI applications  
- API key handling using `.env`  
- Real-world AI system design  

---

# 🙌 Author

## Mohamed Aslam

Passionate about:

- Data Science  
- Generative AI  
- AI Agents  
- Real-world AI products  
- End-to-end AI deployment  

---

# ⭐ Support

If you found this project useful, please give it a **Star ⭐ on GitHub**.

---

# 📜 License

This project is for educational and portfolio purposes.
