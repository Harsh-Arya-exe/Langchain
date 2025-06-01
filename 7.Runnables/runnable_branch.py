from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnablePassthrough, RunnableBranch
from dotenv import load_dotenv
import os
load_dotenv()

hf_key = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
gemini_key = os.getenv("Gemini_Key")

llm = HuggingFaceEndpoint(
    model = "google/gemma-2b-it",
    huggingfacehub_api_token=hf_key
)

model1 = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)

report_gen_chain = RunnableSequence(prompt1, model1, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 300, RunnableSequence(prompt2, model1, parser) ),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

result = final_chain.invoke({'topic': 'Russia vs Ukraine'})

print(result)