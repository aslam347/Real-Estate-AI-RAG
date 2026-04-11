# 🏡 AI Real Estate Intelligence Engine

An AI-powered real estate research assistant built using **Retrieval-Augmented Generation (RAG)**.  
This application allows users to extract insights from real estate articles by simply asking questions.

---

## 🚀 Live Demo
🔗 https://your-streamlit-app-link.streamlit.app

---

## 🎥 Demo Video
🔗 https://your-loom-video-link

---

## 📌 Problem Statement

Analyzing multiple real estate articles manually is time-consuming and inefficient.  
Users struggle to quickly extract key insights like:

- Housing prices 📊  
- Mortgage trends 📉  
- Market comparisons 🏘️  

---

## 💡 Solution

This project provides an **AI-powered assistant** that:

- Reads real estate articles from URLs  
- Stores content in a vector database  
- Retrieves relevant information using semantic search  
- Generates accurate answers using LLM  

---

## 🧠 How It Works (RAG Pipeline)

1. Load article data using `WebBaseLoader`  
2. Split text into chunks  
3. Convert text into embeddings  
4. Store embeddings in **ChromaDB**  
5. Retrieve relevant chunks using similarity search  
6. Pass context to LLM for final answer generation  

---

## ⚙️ Tech Stack

- **LangChain** – orchestration  
- **ChromaDB** – vector database  
- **HuggingFace Embeddings** – semantic search  
- **Groq LLM** – fast inference  
- **Streamlit** – UI & deployment  

---

## ✨ Features

- 🔍 Ask questions from real estate articles  
- ⚡ Fast response using Groq LLM  
- 🧠 Semantic search with embeddings  
- 🔗 Source-based answers (no hallucination)  
- 🎨 Modern UI with cinematic design  

---

## 📦 Installation (Local Setup)

```bash
git clone https://github.com/aslam347/Real-Estate-AI-RAG.git
cd Real-Estate-AI-RAG
pip install -r requirements.txt
