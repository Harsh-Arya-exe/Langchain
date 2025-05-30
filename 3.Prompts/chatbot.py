from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
import os

gemini_key = os.getenv("Gemini_Key")

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=gemini_key
)

chat_history = [
    SystemMessage(content='You are a helpful AI assistant')
]

while True:
    text = input("Enter the message: ")
    chat_history.append(HumanMessage(content=text))
    if(text == 'exit'):
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print(f'\n\nAI: {result.content}\n\n')

print('\n\nFinal Chat Conversation: ', chat_history)