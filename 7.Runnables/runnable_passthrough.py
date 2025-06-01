from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
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

model2 = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=gemini_key
)

prompt1 = PromptTemplate(
    template='Generate a joke about {topic} in 20 words.',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Explain the following joke ->  {topic}',
    input_variables=['topic']
)

joke_gen_chain = RunnableSequence(prompt1, model2, parser)

parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'explanation':RunnableSequence(prompt2, model1, parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({'topic':'cricket'})

print(result)