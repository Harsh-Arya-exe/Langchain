from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()

hf_key = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

llm = HuggingFaceEndpoint(
    model = "google/gemma-2b-it",
    huggingfacehub_api_token=hf_key
)

model = ChatHuggingFace(llm=llm)


class Person(BaseModel):

    name: str = Field(description='Name of the person')
    age: int = Field(gt=18, description='Age of the person')
    city: str = Field(description='Name of the city the person belongs to')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person \n {format_information}',
    input_variables=['place'],
    partial_variables={'format_information':parser.get_format_instructions()}
)

#With Chain
chain = template | model | parser

result = chain.invoke({'place':'Japanese'})

print(result)


"""
Without Chain


prompt = template.invoke({'place':'indian'})

result = model.invoke(prompt)

final_result = parser.parse(result.content)

print(final_result)
"""