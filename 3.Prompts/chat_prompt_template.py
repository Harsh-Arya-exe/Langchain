from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} assistant'),
    ('human', 'Exlpain in 40 words about {topic}')
])

prompt = chat_template.invoke({'domain':'cricket', 'topic':'Dusra'})

print(prompt)

