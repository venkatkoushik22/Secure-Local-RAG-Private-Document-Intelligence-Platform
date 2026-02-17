import os
import shutil
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONFIGURATION ---
DOCS_FOLDER = "./docs"
PERSIST_DIRECTORY = "./chroma_db"  # <--- Where your AI's "memory" is saved
SUPPORTED_EXTENSIONS = {'.py', '.js', '.html', '.css', '.md', '.txt', '.json', '.pdf'}
IGNORE_FOLDERS = {'.git', 'node_modules', 'venv', 'env', '__pycache__', 'build', 'dist'}

def load_documents(folder_path):
    documents = []
    print(f"\nScanning '{folder_path}' for code...")
    for root, dirs, files in os.walk(folder_path):
        dirs[:] = [d for d in dirs if d not in IGNORE_FOLDERS]
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                try:
                    loader = PyPDFLoader(file_path) if ext == ".pdf" else TextLoader(file_path, encoding='utf-8', autodetect_encoding=True)
                    documents.extend(loader.load())
                except Exception: continue
    return documents

# --- MAIN SYSTEM ---
print("\n--- Secure Local RAG: Privacy-First Document Intelligence ---")

# 1. Initialize Embeddings (The "Translator")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 2. Check for Existing Database
if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
    print(f"\n[SUCCESS] Found existing database in '{PERSIST_DIRECTORY}'.")
    print("Loading AI memory from disk... (This is fast!)")
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY, 
        embedding_function=embeddings
    )
else:
    # 3. Create New Database if none exists
    raw_docs = load_documents(DOCS_FOLDER)
    if not raw_docs:
        print("\n[ERROR] No files found to index. Add files to '/docs' and restart.")
        exit()
    
    print(f"Splitting {len(raw_docs)} files into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(raw_docs)
    
    print(f"Indexing {len(chunks)} snippets. Please wait while we save to disk...")
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=PERSIST_DIRECTORY  # <--- Saves it here!
    )

# 4. Setup Retrieval Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=Ollama(model="llama3"),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)



print("\n--- SYSTEM READY! Ask about your code or documents. (Type 'exit' to quit) ---")
while True:
    query = input("\n> ")
    if query.lower() in ['exit', 'quit']: break
    response = qa_chain.invoke(query)
    print(f"\nAI Response:\n{response['result']}")