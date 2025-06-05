from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]


query = "A player having the demeanor as the god"

doc_embeddings = embeddings.embed_documents(documents)
query_embeddings = embeddings.embed_query(query)

scores = cosine_similarity([query_embeddings], doc_embeddings)[0]

print(f"Scores: {scores}")
print('\n')

print("enumerate_Scores: ", enumerate(scores))
print('\n')

print("List of enum scores: ", list(enumerate(scores)))
print('\n')

print("Sorted list of enum scores: ", sorted(list(enumerate(scores))))
print('\n')

index, score = sorted(list(enumerate(scores)), key=lambda x:x[1])[-1]

print(query)
print('\n')
print(documents[index])
print('\n')
print("similarity_scores is: ", score)
print('\n')