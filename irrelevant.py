import os
import requests
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

AZURE_OPEN_AI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPEN_AI_ENDPOINT = os.getenv("ENDPOINT_URL")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

client = AzureOpenAI(
    api_key=AZURE_OPEN_AI_KEY,
    api_version="2024-05-01-preview",
    azure_endpoint=AZURE_OPEN_AI_ENDPOINT
)

SYSTEM_PROMPT = """
You are a campus tour bot assistant. Politely redirect off-topic queries 
to university-related subjects. Use friendly humor and keep responses under 50 words.
Examples:
- "I'm here to talk about campus! Ask me about our library or sports facilities!"
- "Let's focus on campus topics! Want to hear about our engineering building?"
"""


def handle_irrelevant(query):
    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ],
            temperature=0.7,
        )
        reply = response.choices[0].message.content
        print("OpenAI response:", reply)
        return reply

    except Exception as e:
        print("Azure OpenAI SDK Error:", e)
        return None