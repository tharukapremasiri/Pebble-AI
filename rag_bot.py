import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# 1. Configuration
CONNECTION = "postgresql+psycopg://postgres:mysecretpassword@localhost:5432/postgres"
COLLECTION_NAME = "streamlit_pdf_base" # Match app.py for shared data
PDF_PATH = "data/project_info.pdf"

# 2. Initialize Models
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 3. PDF Ingestion
if os.path.exists(PDF_PATH):
    print(f"--- Processing: {PDF_PATH} ---")
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    
    # Tagging metadata with filename
    for page in pages:
        page.metadata["source"] = os.path.basename(PDF_PATH)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)

    vector_store = PGVector.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION,
        use_jsonb=True,
    )
    print(f"--- Successfully indexed {len(docs)} chunks ---")

# 4. Global Search & Strict Chat
query = input("\n🔍 Query all documents: ")
vector_store = PGVector(embeddings=embeddings, collection_name=COLLECTION_NAME, connection=CONNECTION)
results = vector_store.similarity_search(query, k=4)

context_list = [f"[File: {d.metadata.get('source')} Page: {d.metadata.get('page', 0)+1}]\n{d.page_content}" for d in results]
context = "\n\n".join(context_list)

system_msg = "Answer ONLY from context. If not found, say 'This information is not related to the uploaded documents.'"
full_prompt = f"{system_msg}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"

print("\n--- AI Response ---")
response = llm.invoke(full_prompt)
print(response.content)