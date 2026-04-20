import streamlit as st
import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
import tempfile
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# --- PAGE CONFIG ---
st.set_page_config(page_title="Pebble AI", page_icon="🤖", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        /* 1. Reduce main container padding */
        .block-container { 
            padding-top: 20px; 
            padding-bottom: 0rem; 
        }

        /* 2. Sidebar spacing and background */
        section[data-testid="stSidebar"] div.stVerticalBlock {
            padding-top: 10px !important;
            gap: 0.5rem;
        }
        
        [data-testid="stSidebarUserContent"] {
            padding-top: 0.5rem !important;
        }

        /* 3. NEW CHAT BUTTON: Force Dark Blue Background */
        /* This targets the button, its hover state, and its active state */
        div[data-testid="stSidebar"] button[kind="primary"] {
            background-color: #0C083B !important;
            color: white !important;
            border: 1px solid #0C083B !important;
            width: 100%;
        }

        /* Specifically target the hover state which often defaults to red/orange */
        div[data-testid="stSidebar"] button[kind="primary"]:hover,
        div[data-testid="stSidebar"] button[kind="primary"]:active,
        div[data-testid="stSidebar"] button[kind="primary"]:focus {
            background-color: #06386A !important;
            border-color: #06386A !important;
            color: white !important;
        }

        /* 4. SAVED CONVERSATIONS: Text Alignment */
        div[data-testid="stSidebar"] button[kind="secondary"] {
            justify-content: flex-start !important;
            text-align: left !important;
            border: none !important;
            background-color: transparent !important;
            width: 100%;
        }

        div[data-testid="stSidebar"] button[kind="secondary"] div[data-testid="stMarkdownContainer"] p {
            text-align: left !important;
        }

        /* 5. Sidebar Background Color */
        [data-testid="stSidebarContent"] {
            background-color: #1E1F20 !important;
        }
        
        [data-testid="stSidebarContent"] .stMarkdown, 
        [data-testid="stSidebarContent"] h1, 
        [data-testid="stSidebarContent"] h2, 
        [data-testid="stSidebarContent"] h3,
        [data-testid="stSidebarContent"] label {
            color: white !important;
        }
            
    </style>
    """, unsafe_allow_html=True)

# --- 1. CONFIGURATION ---
CONNECTION = "postgresql+psycopg://postgres:mysecretpassword@localhost:5432/postgres"
COLLECTION_NAME = "streamlit_pdf_base"

# --- 2. INITIALIZE MODELS ---
@st.cache_resource
def load_models():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return embeddings, llm

embeddings, llm = load_models()

# --- 3. DATABASE FUNCTIONS FOR HISTORY ---
def get_db_connection():
    return psycopg2.connect("host=localhost dbname=postgres user=postgres password=mysecretpassword")

def save_chat_to_db(title, messages, chat_id=None):
    conn = get_db_connection()
    cur = conn.cursor()
    if chat_id:
        # Update existing session
        cur.execute(
            "UPDATE chat_history SET title = %s, chat_data = %s WHERE id = %s",
            (title, json.dumps(messages), chat_id)
        )
    else:
        # Create new session
        cur.execute(
            "INSERT INTO chat_history (title, chat_data) VALUES (%s, %s)",
            (title, json.dumps(messages))
        )
    conn.commit()
    cur.close()
    conn.close()

def load_all_sessions():
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT id, title FROM chat_history ORDER BY created_at DESC")
        sessions = cur.fetchall()
        cur.close()
        conn.close()
        return sessions
    except Exception:
        return []

# --- 4. SIDEBAR: PDF INGESTION & SESSION MGMT ---
with st.sidebar:
    st.markdown("# 🤖 Pebble AI")
    st.markdown("### *Knowledge Assistant with Memory*")
    
    # --- NEW CHAT OPTION ---
    st.divider()
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_chat_id = None
        st.session_state.chat_title_input = "New Project"
        st.rerun()

    # --- SESSION MANAGEMENT SECTION ---
    st.markdown("### 📚 Saved Conversations")
    sessions = load_all_sessions()
    for s in sessions:
        if st.button(f"💬 {s['title']}", key=f"sess_{s['id']}", use_container_width=True):
            conn = get_db_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute("SELECT id, title, chat_data FROM chat_history WHERE id = %s", (s['id'],))
            data = cur.fetchone()
            st.session_state.messages = data['chat_data']
            st.session_state.current_chat_id = data['id']
            st.session_state.chat_title_input = data['title']
            cur.close()
            conn.close()
            st.rerun()

    st.divider()
    # Initialize chat title in session state if not present
    if "chat_title_input" not in st.session_state:
        st.session_state.chat_title_input = "New Project"
        
    chat_title = st.text_input("Chat Title", value=st.session_state.chat_title_input)
    
    if st.button("💾 Save Conversation", use_container_width=True):
        if st.session_state.get("messages"):
            current_id = st.session_state.get("current_chat_id")
            save_chat_to_db(chat_title, st.session_state.messages, chat_id=current_id)
            st.toast(f"Saved as '{chat_title}'")
            st.rerun() # Refresh to show updated title in list
        else:
            st.warning("No messages to save yet!")

    # --- PDF UPLOAD SECTION ---
    st.divider()
    st.header("Upload Documents")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if uploaded_file and st.button("Index Document"):
        with st.status("Processing PDF...", expanded=True) as status:
            st.write("📂 Preparing file...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            st.write("📄 Reading and splitting...")
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            for page in pages:
                page.metadata["source"] = uploaded_file.name
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = text_splitter.split_documents(pages)
            
            st.write("🐘 Syncing to PostgreSQL...")
            PGVector.from_documents(
                documents=docs,
                embedding=embeddings,
                collection_name=COLLECTION_NAME,
                connection=CONNECTION,
                use_jsonb=True,
            )
            os.remove(tmp_path)
            status.update(label=f"✅ {uploaded_file.name} added!", state="complete", expanded=False)

# --- 5. MAIN CHAT INTERFACE ---
st.title("📚 Pebble Knowledge Hub")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask about your uploaded PDFs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            vector_store = PGVector(embeddings=embeddings, collection_name=COLLECTION_NAME, connection=CONNECTION)
            results = vector_store.similarity_search(prompt, k=4)
            
            context_text = ""
            if results:
                context_list = []
                for doc in results:
                    source = doc.metadata.get("source", "Unknown Document")
                    page = doc.metadata.get("page", 0) + 1 
                    context_list.append(f"--- SOURCE: {source} (Page {page}) ---\n{doc.page_content}")
                context_text = "\n\n".join(context_list)
            
            history_context = ""
            for m in st.session_state.messages[-5:-1]:
                history_context += f"{m['role'].capitalize()}: {m['content']}\n"

            system_message = (
                "You are a professional assistant. Answer the question using the provided context. "
                "Use the 'Past Conversation' to understand follow-up questions. "
                "If the answer is in the context, summarize it and mention the source/page. "
                "If not in context, say: 'I'm sorry, that info isn't in the uploaded documents.'"
            )
            
            final_prompt = (
                f"{system_message}\n\n"
                f"Past Conversation:\n{history_context}\n"
                f"PDF Context:\n{context_text}\n\n"
                f"User Question: {prompt}\n"
                f"Assistant Answer:"
            )
            
            response = llm.invoke(final_prompt)
            response_text = response.content
            
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})