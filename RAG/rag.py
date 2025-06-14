from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain.schema import Document
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

hf_key = os. getenv("HUGGINGFACEHUB_ACCESS_TOKEN")


llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2b-it",
    huggingfacehub_api_token=hf_key
)

#ChatLLM
model = ChatHuggingFace(llm=llm)
#embedding model
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

parser = StrOutputParser()


#1.Indexing
#1.a Data Ingestion
loader = PyPDFLoader('Document_loaders\\dl-curriculum.pdf')

docs = loader.load()

documents = []

for i in range(len(docs)):
    print(f"Docs no. {i}\n{docs[i].page_content}\n")
    documents.append(docs[i].page_content)

#1.b Chunking
splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type='standard_deviation',
    breakpoint_threshold_amount=1
)

chunks = splitter.create_documents(documents)

chunk_list = []
print("Lenght: ", len(chunks))
for i in range(len(chunks)):
    print(f"Chunk no. {i}\n{chunks[i].page_content}\n")
    chunk_list.append(chunks[i].page_content)
    
#1.c Embedding
#chunk_embeddings = embeddings.embed_documents(chunk_list)

#Here the embeddings are done in the Vector Store so i commented this line

#1.4 Vector Store
vector_store = Chroma(
    embedding_function=embeddings,
    collection_name='Chroma_sample'
)

#vec_document = [Document(page_content=chunk) for chunk in chunk_list]
#vector_store.add_documents(vec_document)

#instead of doing this we can also do this
vector_store.add_documents(chunks) 

#2.Retrieval
retriever = vector_store.as_retriever(
    search_type='mmr',
    search_kwargs={'k':3, 'lambda_mult':0.5}
)

query = "Is Fine tuning Present in the syllabus?"

valid_chunks = retriever.invoke(query)

print("Valid_chunks: ", valid_chunks)

context = [doc.page_content for doc in valid_chunks]

print("Context: ", context)

#3.Augumentation
prompt = PromptTemplate(
    template = """
    You are a helpful assistant. Answer the following query asked by user from the given context. If you the context is not sufficient then 
    just reply with I don't know followed by an apology

    {context}
    query: {query}
    """,
    input_variables=['context', 'query']
)
#4.Generation i guess

#formatted_prompt = template.format(
    #context=context,  # from previous step
    #query=query
#)

chain = prompt | model | parser

# 3. Generate response
response = chain.invoke({'context':context, 'query':query})

print("\nResponse:", response)


