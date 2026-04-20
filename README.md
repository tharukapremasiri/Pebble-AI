# 🤖 Pebble AI: Persistent Knowledge Assistant

Pebble AI is a sophisticated RAG (Retrieval-Augmented Generation) application that transforms static PDF documents into a dynamic, searchable, and conversational knowledge base. Built with a focus on source transparency and persistent memory, it enables seamless research across multiple sessions.



## 🚀 Key Features

- **Multi-Document RAG:** Ingest and query across multiple PDF files simultaneously.
- **Source Transparency:** Every AI response includes citations referencing the specific source file and page number.
- **Persistent Chat History:** Save and resume your conversations. All chat titles and message history are stored in a PostgreSQL database.
- **Conversational Memory:** Remembers past exchanges within a session to handle follow-up questions effectively.
- **Customized UI:** A streamlined Streamlit interface with a dark-mode theme, left-aligned conversation history, and an intuitive sidebar.
- **Dockerized Vector Storage:** Utilizes `PGVector` on PostgreSQL for high-performance vector similarity searches.

## 🛠️ Technical Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **Orchestration:** [LangChain](https://www.langchain.com/)
- **LLM:** OpenAI GPT-4o-mini
- **Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- **Database:** PostgreSQL with `PGVector` extension
- **Driver:** `psycopg2-binary` for session management
- **Infrastructure:** Docker

## 📋 Prerequisites

- Python 3.10+
- Docker Desktop
- OpenAI API Key

## 🔧 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/tharukapremasiri/Pebble-AI.git](https://github.com/tharukapremasiri/Pebble-AI.git)
   cd Pebble-AI

2. **Set up a Virtual Environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # source venv/bin/activate  # On macOS/Linux
3.**Install dependencies:**
  ```bash
  pip install streamlit langchain langchain-openai langchain-community langchain-postgres langchain-huggingface psycopg2-binary pypdf sentence-transformers
  ```
4.**Spin up the Database (Docker):**
  ```bash
  docker run --name rag-postgres -e POSTGRES_PASSWORD=mysecretpassword -p 5432:5432 -d ankane/pgvector
  ```
5.**Spin up the Database (Docker):**
```bash
  docker run --name rag-postgres -e POSTGRES_PASSWORD=mysecretpassword -p 5432:5432 -d ankane/pgvector
  ```
6.**Run the Application:**
```bash
  streamlit run app.py
  ```

## 📝 Usage
- **Upload Docs:** Use the sidebar to upload PDF files. Click Index Document to process them.
- **Chat:** Ask questions about your documents. Pebble AI will provide answers with page-level citations.
- **Save/Load:** Give your chat a title and hit Save Conversation. You can reload it later from the Saved Conversations list.
- **New Chat:** Use the ➕ New Chat button to clear the current window and start a fresh session.

## 👤 Author: Tharuka Premasiri 
- Developed by Tharuka Premasiri

