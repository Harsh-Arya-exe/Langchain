from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

hf_key = os. getenv("HUGGINGFACEHUB_ACCESS_TOKEN")


llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2b-it",
    huggingfacehub_api_token=hf_key
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

#parser2 = StrOutputParser()

template = PromptTemplate(
    template='Give me the name, age and city of a fictional character \n {format_instruction}',
    input_variables=[''],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# prompt = template.format()    this line return a prompt which is supported by model.invoke
#                               it would be like this -> Give me the name, age and city of a fictional character 
# Return a JSON format

#WITH CHAIN

chain =  template | model | parser

result = chain.invoke({})

print(result)
"""
WITHOUT CHAIN


result = model.invoke(prompt)
 
final_result = parser.parse(result.content)

print(final_result)
"""