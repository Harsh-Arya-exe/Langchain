from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda
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

prompt = PromptTemplate(
    template='Generate a joke about {topic} in 20 words.',
    input_variables=['topic']
)

joke_gen_chain = RunnableSequence(prompt, model2, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(lambda x : len(x.split())) # this is one way to write RunnableLambda
})

#Other way is 
"""
def word_count(text):
    return len(text.split())

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count) # this is another way to write RunnableLambda
})

"""

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({'topic':'AI'})

final_result = """{} \nword count - {}""".format(result['joke'], result['word_count'])

print(final_result)