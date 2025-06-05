from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

gemini_key = os.getenv("Gemini_Key")


model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=gemini_key
)

template = PromptTemplate(
    template="Greet this person in 5 languages. The name of the person is {name}",
    input_variables=['name']
)

prompt = template.invoke({'name' : 'Harsh'})

result = model.invoke(prompt)

print(result.content)
