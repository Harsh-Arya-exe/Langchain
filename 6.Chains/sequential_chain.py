from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']

)

prompt2 = PromptTemplate(
    template='Give a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"topic":"Hulk"})

print(f'{result}')
