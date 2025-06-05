from langchain_chroma import Chroma
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv 
import os

load_dotenv()

hf_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")


# Step 1: Your source documents
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

# Step 2: Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

# Step 3: Create Chroma vector store in memory
vector_store = Chroma(
    embedding_function=embeddings,
    collection_name='my_collection'
)

vector_store.add_documents(documents=documents)

# Step 4: Convert vectorstore into a retriever
retriever = vector_store.as_retriever(kwargs={'k':2})

query = "What is Chroma used for?"

result = retriever.invoke(query)

for i, doc in enumerate(result):
    print(f"\n--- Result {i+1} ---")
    print(f"Content:\n{doc.page_content}...")  # truncate for display