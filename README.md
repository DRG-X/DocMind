DocMind
### *Talk to your documents. Local, private, and blazing fast.*

[![Python](https://img.shields.io/badge/Python-%233776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-%23FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-green?style=for-the-badge)](https://ollama.ai)

---

## 🧩 Overview

**Chat with PDFs** lets you upload one or multiple PDFs and chat with them — locally, privately, and efficiently.  
It extracts text, chunks and embeds it for vector search, then uses an LLM (local or cloud) to respond with context-aware answers.

Perfect for:
- 📚 Researchers summarizing papers  
- 🧠 Students studying long notes  
- 🏢 Analysts or teams with confidential reports  

---

## ⚙️ Core Features  

### 🧠 PDF Text Extractor  
Reads and cleans text from any PDF — the core engine that enables intelligent chat.  
> *Lit because it turns dead static files into readable data for AI.*

### 🧩 Text Chunking & Embeddings  
Splits large documents into small chunks and converts them into embeddings for efficient recall.  
> *Lit because it lets the AI “remember” and “understand” long PDFs efficiently.*

### 🗂️ Vector Database (FAISS / Chroma)  
Stores all chunks so relevant parts can be fetched instantly.  
> *Lit because it makes search and context retrieval blazing fast.*

### 💬 LLM Query Engine  
Handles user questions and crafts responses using both PDF + vector database context.  
> *Lit because it feels like you’re literally talking to your document.*

### 🖥️ Streamlit / Gradio Interface  
Minimal, clean web interface for file uploads and chat interaction.  
> *Lit because it looks professional while staying minimal and fast to deploy.*

---

## ⚡ Advanced Features  

### 📚 Multi-PDF Support  
Upload and query across multiple PDFs seamlessly.  
> *Lit because it mimics real-world workflows — researchers, analysts, students.*

### 🔒 Offline Mode (Local LLM / Ollama)  
Runs fully offline with local models for privacy and speed.  
> *Lit because no data leaves your device — perfect for research or enterprise users.*

### 🧠 Context Memory  
Maintains chat history for coherent conversations.  
> *Lit because it makes the AI feel like it actually understands the convo thread.*

### 🧾 Smart Summary Button  
One-click instant TL;DR generation for the entire PDF.  
> *Lit because you can process huge reports or papers in seconds.*

### 💾 Export Responses  
Save or share chat transcripts and summaries as TXT or PDF.  
> *Lit because it makes the tool practical, not just cool.*

---

## 💎 Future Add-ons (for scaling later)  

- ☁️ **Drive Integration:** Pull PDFs directly from Google Drive or Dropbox.  
  > *Lit because automation = convenience = user retention.*

- 🤝 **Collab Mode:** Share chats with teammates or classmates.  

- 🧩 **Multi-LLM Support:** Switch between local Ollama, OpenAI, Claude, or Gemini.  

- 🔍 **Citation Mode:** Include inline references with quoted PDF page numbers.  

---

## 🧠 Tech Stack

| Layer | Tools |
|-------|-------|
| **Frontend** | Streamlit / Gradio |
| **Vector DB** | FAISS / Chroma |
| **Embedding Model** | Sentence Transformers / OpenAI |
| **LLM Backend** | Ollama (local) / OpenAI API |
| **Text Processing** | PyPDF2 / pdfplumber |
| **Deployment** | Streamlit Cloud / local runtime |

---
## 👨‍💻 Author

**Deepansh Raj Goel**  
💼 Student | AI & Automation Developer | Builder of useful tech  
🔗 [LinkedIn]((https://www.linkedin.com/in/deepansh-raj-goel-416b77378/))) • [GitHub](https://github.com/DRG-X))
## ⚙️ Setup  

```bash
# clone
git clone https://github.com/<your-username>/Chat-with-PDFs.git
cd Chat-with-PDFs

# install deps
pip install -r requirements.txt

# run locally
streamlit run app.py





