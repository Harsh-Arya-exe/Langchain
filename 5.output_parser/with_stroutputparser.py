from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
load_dotenv() 

hf_key = os. getenv("HUGGINGFACEHUB_ACCESS_TOKEN")


llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2b-it",
    task='text-generation',
    huggingfacehub_api_token=hf_key
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template="Write a detailed report on {topic}.",
    input_variables=['topic']
)

template2 = PromptTemplate(
    template="Write a 5 line summary on the following text. \n {text}.",
    input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'black hole'})

print(result)