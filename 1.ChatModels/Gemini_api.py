from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

gemini_key = os.getenv("Gemini_Key")

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=gemini_key
)

result = model.invoke("What is the capital of India?")

print(result.content)