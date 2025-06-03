# All the document loader will be in the langchain_community
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda
from dotenv import load_dotenv
import os

load_dotenv() 

hf_key = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

llm = HuggingFaceEndpoint(
    model = "google/gemma-2b-it",
    huggingfacehub_api_token=hf_key
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt = PromptTemplate(
    template='Write a summary of the following poem - {poem}',
    input_variables=['poem']
)

loader = TextLoader('D:\Langchain\8.Document_loaders\cricket.txt', encoding='utf-8')

docs = loader.load()

print(docs)

print(type(docs)) # docs is a list 

print(docs[0]) # docs[0] because it had only one element

print(type(docs[0]))

topic = RunnableLambda(lambda _: docs[0].page_content) # using RunnableLambda

chain = topic | prompt | model | parser

result = chain.invoke({}) # could also be done like {'topic':docs[0].page_content}

print("\nSummary: ", result)