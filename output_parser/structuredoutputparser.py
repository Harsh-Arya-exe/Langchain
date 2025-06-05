from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate
# StructuredOutputParser is in langchain unlike other parser which are in langchain_core

load_dotenv()

hf_key = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

llm = HuggingFaceEndpoint(
    model = "google/gemma-2b-it",
    huggingfacehub_api_token=hf_key
)

model = ChatHuggingFace(llm=llm)

# First we have to make a Schema , made with ResponseSchema class where we send list of the list of schema objs

schema = [
    # Schema object
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'), 
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic')
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Give 3 facts about the {topic} \n {format_instruction}" ,
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)


prompt = template.invoke({'topic':'Black Hole'})

result = model.invoke(prompt)

final_result = parser.parse(result.content)

print(final_result)

# Downside of StroutputParser -> No data Validation
# Hence the PydanticOutputParser



