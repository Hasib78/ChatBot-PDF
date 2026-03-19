# AI-Powered PDF Chatbot (RAG System)

A full-stack Retrieval-Augmented Generation (RAG) application that allows users to upload multiple PDF documents and converse with their data in real-time. 

## 🚀 Live Demo
* [Frontend Application](Link-to-your-streamlit-app)
* [Backend API Docs](Link-to-your-fastapi-docs)

## 🏗️ Architecture
* **Frontend:** Streamlit (Provides an interactive, chat-like UI with file management)
* **Backend:** FastAPI (Handles asynchronous document processing and API routing)
* **AI/NLP:** LangChain, HuggingFace Embeddings (`all-MiniLM-L12-v2`)
* **Vector Database:** ChromaDB (Locally persisted for document retrieval)

## 💡 Key Features
* **Multi-Document Processing:** Upload and parse multiple PDFs simultaneously.
* **Smart Chunking:** Utilizes RecursiveCharacterTextSplitter for optimal context retrieval.
* **State Management:** Fully functional chat history with dynamic source citing.
* **Memory Wipe:** Custom endpoint to safely release OS file locks and purge the vector database.



<!-- Project Title: "RAG_PDFChatbot"

Setup Instructions: 
Create Virtual Environment: #python -m venv myenv
python -m venv venv

Activate myenv : venv\Scripts\activate
pip install -r requirements.txt

Environment Variables: A note to add the GROQ_API_KEY to the .env file.

How to Run: fastapi dev server/main.py
fastapi dev main.py
streamlit run app.py 
Deploy: uvicorn main:app --host 0.0.0.0 --port 10000-->