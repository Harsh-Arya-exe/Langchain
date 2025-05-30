from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()

print("HF_TOKEN: ", os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"))


hf_key = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

client = InferenceClient(
    provider="novita",
    api_key=hf_key,
)

completion = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of India?"
        }
    ],
)

print(completion.choices[0].message.content)