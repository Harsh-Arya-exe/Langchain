from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from dotenv import load_dotenv
import os

load_dotenv()

hf_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory='chroma_db',
    collection_name='sample'
)

# Create LangChain documents for IPL players

doc1 = Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
        metadata={"team": "Royal Challengers Bangalore"}
    )
doc2 = Document(
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
        metadata={"team": "Mumbai Indians"}
    )
doc3 = Document(
        page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
        metadata={"team": "Chennai Super Kings"}
    )
doc4 = Document(
        page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
        metadata={"team": "Mumbai Indians"}
    )
doc5 = Document(
        page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
        metadata={"team": "Chennai Super Kings"}
    )

docs = [doc1, doc2, doc3, doc4, doc5]

#add documents
vector_store.add_documents(docs)

#view documents
vector_store.get(include=['embeddings', 'documents', 'metadatas'])

#search_documents 
vector_store.similarity_search(
    query='Who among these is a batsman?',
    k=2
)

#search with similarity score
vector_store.similarity_search_with_score(
    query='Who among these is a batsman?',
    k=2
)

#meta-data filtering
vector_store.similarity_search(
    query = '',
    filter = {'team':'Royal Challengers Bangalore'}
)


#update document
updated_doc1 = Document(
    page_content = "Virat Kohli, the former captain of Royal Challengers Bangalore (RCB), is renowned for his aggressive leadership and consistent batting performances. He holds the record for the most runs in IPL history, including multiple centuries in a single season. Despite RCB not winning an IPL title under his captaincy, Kohli's passion and fitness set a benchmark for the league. His ability to chase targets and anchor innings has made him one of the most dependable players in T20 cricket.",
    metadata = {'team':'Royal Challengers Bangalore'}
)
vector_store.update_document(
    document_id = '25a86b43-58a6-4f80-a3e6-c6d8c3c4db35',
    document = updated_doc1
)
vector_store.get(include=['embeddings', 'documents', 'metadatas'])

#delete document
vector_store.delete(
  ids= '9eeb506d-2679-4ceb-b4eb-cb7ad56df4d8'
)
vector_store.get(include=['embeddings', 'documents', 'metadatas'])
