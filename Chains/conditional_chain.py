from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
import os

load_dotenv()

hf_key = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
gemini_key = os.getenv("GEMINI_KEY")

llm = HuggingFaceEndpoint(
    model = "google/gemma-2b-it",
    huggingfacehub_api_token=hf_key
)

model = ChatHuggingFace(llm=llm)

parser1 = StrOutputParser()

class Feedback(BaseModel):

    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template="""You are a customer service agent. The following is a positive feedback from a customer. 
Respond with appreciation and maintain a professional tone.

Feedback: "{feedback}" """,
    input_variables=['feedback']
)

prompt3 =  PromptTemplate(
    template="""You are a customer support agent. The following is a negative customer feedback. 
Respond to the customer in a polite, professional, and empathetic manner. Do not ask for more details, just respond appropriately.

Feedback: "{feedback}" """,
    input_variables=['feedback']
)

#Branch Chain(if - else) 
#in this we give multiple tuples 

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | parser1),
    (lambda x: x.sentiment == 'negative', prompt3 | model | parser1),
    RunnableLambda(lambda x: "could not find sentiment")
)

final_chain = classifier_chain | branch_chain

result = final_chain.invoke({'feedback': 'This is a wonderful phone'})

print("Result: ",result)