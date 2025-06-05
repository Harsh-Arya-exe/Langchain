from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, load_prompt
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

gemini_key = os.getenv("Gemini_Key")

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=gemini_key
)

st.header("Research Tool")

paper_input = st.selectbox(
    "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] 
)

length_input = st.selectbox(
     "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] 
)

style_input = st.selectbox(
    "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] 
)

template = load_prompt('template.json')


if st.button('Summarize'):

    chain = template | model
    result = chain.invoke(
    {
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
    }
    )
    st.write(result.content)

