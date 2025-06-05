from langchain_community.vectorstores import FAISS
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv 
import os

load_dotenv()

hf_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

vector_store = FAISS.from_documents(
    documents=docs,
    embedding=embeddings
)

retriever = vector_store.as_retriever(
    search_type = 'mmr',
    search_kwargs={'k':3, 'lambda_mult':0.5}
)

query = "What is Langchain?"

result = retriever.invoke(query)

for i, doc in enumerate(result):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)